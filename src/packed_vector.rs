use std::cmp::Ordering;

/// A sparse vector may be held in a full-length vector of storage.
/// But to economize in storage, we may pack the vector by holding the entries as real, interger
/// pairs. Here we implement this idea by using a f64 array to store the data, and a usize array to
/// store the index.
///
/// In general, it is easy to see that the packed form requires less storage when the
/// vector is at least 50% sparse. When the numerical values in the vector are held
/// in double-precision arrays, or if the values are single-precision complex numbers,
/// then the break-even point will drop to 25%, and will drop even further in the
/// extended-precision real or double-precision complex case. Thus, the packed form
/// generally requires far less storage in practical computations, where the vectors,
/// at least at the beginning of the computation, are far less dense than 25%.
#[derive(Clone, Debug)]
pub struct PackedVec {
    /// Store the index of the non-zero data
    index: Vec<usize>,
    /// Store the non-zero data
    data: Vec<f64>,

    /// Store the original vector length for easier scatter back
    full_length: usize,
}

impl Default for PackedVec {
    fn default() -> Self {
        Self::new()
    }
}

impl PackedVec {
    /// Create a empty PackedSparseVec to represent a sparse vector.
    pub fn new() -> Self {
        Self {
            index: Vec::new(),
            data: Vec::new(),
            full_length: 0,
        }
    }

    /// Gather is a special verb describing a transformation from full-length array to a packed
    /// sparse vector.
    pub fn gather(original: &[f64]) -> Self {
        let (index, data): (Vec<usize>, Vec<f64>) = original
            .iter()
            .enumerate()
            .filter(|(_, &x)| x != 0.0)
            .unzip();

        let is_dense = original.len() <= (index.len() + data.len());
        if is_dense {
            println!(
                "Warning: the original array is dense, `gather` it will require a bigger storage."
            )
        }

        Self {
            index,
            data,
            full_length: original.len(),
        }
    }

    /// Scatter is a special verb describing the transformation from packed vector to full-length
    /// array.
    pub fn scatter(&self) -> Vec<f64> {
        let mut full_len_v = vec![0.0; self.full_length];

        for kx in 0..self.len() {
            let ix = self.index[kx];
            full_len_v[ix] = self.data[kx];
        }

        full_len_v
    }

    /// Return the amount of the non-zero component
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return true if all component are zero
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Adding a multiple of one vector to another. To distinguish the index of the packed vector
    /// and the actual index of the full-length vector, I use `k` denote that it is the index of
    /// packed vector and `i` to denote that it is the index of the actual vector.
    ///
    /// For example, `ix` is a index for the full-length vector X and `kx` is a index for the
    /// packed vector.
    pub fn mul_add(&mut self, y_vec: &Self, alpha: f64) {
        // Use a helper storage for flagging non-zero entry in the Y vector.
        //
        // There are two approch.
        //  * One is using the full-length vector for all time manipulation. This is simpler
        //  implementation.
        let mut tmp = vec![None; self.full_length];

        //  * The second one is to use a vector only spanning from the first entry to the second
        //  entry. In some case, this can save much space comparing to the full-length vector.
        //  TODO: Implement a shrink vector for holding temporary index entry.

        // #1: Store the y_vec data index

        // for each entry Y[k], place its position in the vector tmp[k].
        for ky in 0..y_vec.len() {
            let iy = y_vec.index[ky];
            tmp[iy] = Some(ky);
        }

        // #2: For existing entry in x_vec, multiply the y vector and add it into x.

        // Scan the X vector, for each entry X[i], check tmp[i] to get the packed vector index Y_k.
        // If the index is not None, use it to find the value of Y[i], and reset tmp[k] to None.
        for kx in 0..self.len() {
            let ix = self.index[kx];
            if let Some(ky) = tmp[ix].take() {
                self.data[kx] += alpha * y_vec.data[ky];
            }
        }

        // #3: For unexisting entry in x_vec, fill the empty place

        // Scan the Y vector. For each entry Y[i] check tmp[i]. If it is not None, add a new
        // component with value `Alpha * Y[i]` to the packed form of X. Reset tmp[i] to None.
        for ky in 0..y_vec.len() {
            let iy = y_vec.index[ky];
            if tmp[iy].take().is_some() {
                self.data.push(alpha * y_vec.data[ky]);
                self.index.push(iy);
            }
        }
    }
}

impl std::ops::Mul for PackedVec {
    type Output = f64;

    /// Inner product of two packed vectors
    fn mul(self, rhs: Self) -> Self::Output {
        let mut product = 0.0;
        let mut kx = 0;
        let mut ky = 0;

        loop {
            if kx == self.len() || ky == rhs.len() {
                break;
            }

            let ix = self.index[kx];
            let iy = rhs.index[ky];
            match ix.cmp(&iy) {
                Ordering::Equal => {
                    product += self.data[kx] * rhs.data[ky];
                    kx += 1;
                    ky += 1;
                }
                Ordering::Greater => {
                    ky += 1;
                }
                Ordering::Less => {
                    kx += 1;
                }
            }
        }

        product
    }
}

#[test]
fn test_packed_vector() {
    #[rustfmt::skip]
    let x = vec![
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 2.0,
        0.0, 0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0,
    ];

    let mut packed_x = PackedVec::gather(&x);
    assert_eq!(packed_x.full_length, x.len());
    assert_eq!(packed_x.data, [1.0, 2.0, 3.0, 1.0]);
    assert_eq!(packed_x.index, [6, 9, 12, 18]);
    assert!(!packed_x.is_empty());
    assert_eq!(packed_x.len(), 4);

    #[rustfmt::skip]
    let y = vec![
        0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 7.0, 2.0,
        0.0, 0.0, 2.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0,
    ];

    let packed_y = PackedVec::gather(&y);

    let inner_product = packed_x.clone() * packed_y.clone();
    assert_eq!(inner_product, 11.0);

    packed_x.mul_add(&packed_y, 32.0);
    assert_eq!(
        packed_x.data,
        [
            1.0 + 1.0 * 32.0,
            2.0 + 2.0 * 32.0,
            3.0 + 2.0 * 32.0,
            1.0,
            1.0 * 32.0,
            7.0 * 32.0,
            1.0 * 32.0
        ]
    );

    let scatter_back = packed_y.scatter();
    assert_eq!(y, scatter_back);
}
