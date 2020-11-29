use itertools::Itertools;

// Reverse Stage 3 by returning (i, j) such that
//  c = confusion[input[i * 2] as usize] ^ confusion[input[i * 2 + 1] as usize + 256_usize]
//                \___ i ___/                        \_____ j ______/
pub fn xor_match(confusion: &[u8;512], char: u8) -> Vec<(usize, usize)> {
    let mut res: Vec<(usize, usize)> = Vec::new();
    for i in 0..256 {
        for j in 0..256 {
            // Because confusion is a lookup table for characters (8 bits), i and j must be
            // between 0 and 256. During stage 3, j (a character) is added with 256 giving a new
            // entry in confusion.
            if confusion[i] ^ confusion[j + 256] == char {
                res.push((i, j))
            }
        }
    }
    return res;
}

// Stage 2 shows a matrix multiplication-like operation. Returned 32x32 matrix where row i column j
// holds `(diffusion[i] >> j) & 1`. Matrix multiplication in stage 2 uses bitwise-xor as vector addition
pub fn compute_matrix(diffusion: &[u32; 32]) -> Vec<Vec<u8>> {
    let mut res: Vec<Vec<u8>> = vec![vec![0;32];32];
    for i in 0..32 {
        for j in 0..32 {
            res[i][j] = ((diffusion[i] >> j) & 1) as u8;
        }
    }
    return res;
}

pub fn matrix_mult(mat: &Vec<Vec<u8>>, vec: &Vec<u8>) -> Vec<u8> {
    let mut res: Vec<u8> = vec![0; vec.len()];
    for i in 0..res.len() {
        let mut s = 0;
        for j in 0..vec.len() {
            s ^= mat[i][j] * vec[j];
        }
        res[i] = s;
    }
    return res;
}

// Swaps row i and row j in matrix
fn swap(matrix: &mut Vec<Vec<u8>>, i: usize, j: usize) {
    matrix.swap(i, j);
}

// Keep row i the same but "add" (bitwise xor) row i to row j and set that to row j
fn add(matrix: &mut Vec<Vec<u8>>, i: usize, j: usize) {
    let new_row: Vec<u8> = matrix[j].iter().zip(matrix[i].iter()).map(|(nj, ni)| {
        nj ^ ni
    }).collect();
    matrix[j] = new_row;
}

// Perform row operations on the matrix to get into rref form. This is essentially solving
// a system of equations involving xors.
pub fn compute_inverse(matrix: &Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    // Make identity matrix and reflect all row operations on it
    let mut res: Vec<Vec<u8>> = vec![vec![0;32];32];
    for i in 0..32 {
        res[i][i] = 1;
    }
    // Make a deep copy of matrix
    let mut current: Vec<Vec<u8>> = matrix.iter().map(|v| v.clone()).collect();
    for i in 0..32 {
        let pivot = current[i][i];
        if pivot == 0 {
            // Scan down for a row with a 1 as pivot
            for j in (i+1)..32 {
                if current[j][i] == 1 {
                    swap(&mut current, i, j);
                    swap(&mut res, i, j);
                    break;
                }
            }
        }
        // Pivot is now a 1
        // Eliminate down
        for j in (i+1)..32 {
            if current[j][i] == 1 {
                add(&mut current, i, j);
                add(&mut res, i, j);
            }
        }
        // Eliminate up
        for j in 0..i {
            if current[j][i] == 1 {
                add(&mut current, i, j);
                add(&mut res, i, j);
            }
        }
    }
    return res;
}

// Given some value from confusion, find a character whose substitution is the given value.
// There 256 different values in confusion as confusion holds 8 bit integers.
pub fn build_lookup_table(confusion: &[u8;512]) -> Vec<Vec<u8>> {
    let mut res = Vec::new();
    for _ in 0..256 {
        res.push(Vec::new());
    }
    for c in 0..256 {
        res[confusion[c] as usize].push(c as u8);
    }
    return res
}

// Takes a list of targets and attempts to reverse stage 2 and stage 1 yielding a new list. Each
// element of this new list will be an "input" that after moving through stage 1 and stage 2 will
// be in the given list. We are essentially computing a preimage on the given list.
pub fn reverse_targets(inverse: &Vec<Vec<u8>>, lookup: &Vec<Vec<u8>>, target_vectors: &Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut res = Vec::with_capacity(target_vectors.len());
    for target in target_vectors.iter() {
        let preimage = matrix_mult(inverse, target);
        // ith element will be set of characters that substitutes into preimage[i]
        let possible_inputs_product: Vec<Option<Vec<u8>>> = preimage.iter().map(|n| {
            let characters: &Vec<u8>  = &lookup[*n as usize];
            if characters.is_empty() {
                None
            } else {
                Some(characters.clone())
            }
        }).collect();
        if !possible_inputs_product.iter().any(|op| op.is_none()) {
            let possibility_iter = possible_inputs_product.iter().map(|op| op.as_ref().unwrap().iter().map(|c| *c)).multi_cartesian_product();
            for possibility in possibility_iter {
                res.push(possibility);
            }
        }
    }

    return res;
}

pub fn crack(target: &[u8;32], diffusion: &[u32;32], confusion: &[u8; 512], rounds: usize) -> Vec<Vec<u8>> {
    let matrix = compute_matrix(&diffusion);
    let inv = compute_inverse(&matrix);
    let lookup = build_lookup_table(&confusion);

    // Reverse stage 3 ... find vector that when passes through stage 3 yields the target
    let mut final_possibilities = Vec::new();
    for c in &target[0..16] {
        let matches = xor_match(&confusion, *c);
        final_possibilities.push(matches);
    }
    // Iterate over every single possible element in stage 3 "preimage"
    for possibility in final_possibilities.iter().map(|v| v.iter()).multi_cartesian_product() {
        // Map `possibility` to a contiguous byte vector that will yield the target after passing through
        // stage 3
        let mut possibility_vec = vec![0_u8; 32];
        for (idx, (i, j)) in possibility.iter().enumerate() {
            possibility_vec[2*idx] = *i as u8;
            possibility_vec[2*idx + 1] = *j as u8;
        }

        // Reverse stage 2 and stage 1 repeatedly
        let mut target_vectors = vec![possibility_vec];
        for _ in 0..rounds {
            target_vectors = reverse_targets(&inv, &lookup, &target_vectors);
        }
        // If the preimage is empty, there was no match. Continue to another final vector
        if !target_vectors.is_empty() {
            return target_vectors;
        }
    }

    return Vec::new();
}

