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
#[derive(Clone)]
pub struct PackedSparseVec {
    /// Store the index of the non-zero data
    index: Vec<usize>,
    /// Store the non-zero data
    data: Vec<f64>,

    /// Store the original vector length for easier scatter back
    full_length: usize,
}

impl Default for PackedSparseVec {
    fn default() -> Self {
        Self::new()
    }
}

impl PackedSparseVec {
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
    /// For example, `x_i` is a index for the full-length vector X and `x_k` is a index for the
    /// packed vector.
    pub fn mul_add(&mut self, y_vec: Self, alpha: f64) {
        let mut tmp = Vec::with_capacity(self.full_length);

        // #1: Store the y_vec data index

        // for each entry Y[k], place its position in the vector tmp[k].
        for k in 0..y_vec.len() {
            let y_i = y_vec.index[k];
            tmp[y_i] = Some(k);
        }

        // #2: For existing entry in x_vec, multiply the y vector and add it into x.

        // Scan the X vector, for each entry X[i], check tmp[i] to get the packed vector index Y_k.
        // If the index is not None, use it to find the value of Y[i], and reset tmp[k] to None.
        for k in 0..self.len() {
            let x_i = self.index[k];
            if let Some(y_k) = tmp[x_i].take() {
                self.data[k] += alpha * y_vec.data[y_k];
            }
        }

        // #3: For unexisting entry in x_vec, fill the empty place

        // Scan the Y vector. For each entry Y[i] check tmp[i]. If it is not None, add a new
        // component with value `Alpha * Y[i]` to the packed form of X. Reset tmp[i] to None.
        for k in 0..y_vec.len() {
            let y_i = y_vec.index[k];
            if tmp[y_i].take().is_some() {
                self.data.push(alpha * y_vec.data[k]);
                self.index.push(y_i);
            }
        }
    }
}