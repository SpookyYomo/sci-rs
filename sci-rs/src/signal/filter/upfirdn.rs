pub(self) mod apply {
    use core::ops::Neg;
    use ndarray::ArrayView1;
    use num_traits::Num;

    pub(super) enum Mode {
        Constant = 0,
        Symmetric = 1,
        ConstantEdge = 2,
        Smooth = 3,
        Periodic = 4,
        Reflect = 5,
        Antisymmetric = 6,
        Antireflect = 7,
        Line = 8,
    }

    // Generic allows for array, ndarray and vec by ref and value.
    pub(super) fn extend_left<'a, T, A>(x: A, idx: isize, mode: Mode, cval: T) -> T
    where
        T: Num + Neg<Output = T> + Copy + 'a,
        isize: Into<T>,
        A: Into<ArrayView1<'a, T>>,
    {
        let x = x.into();
        let len_x = x.len() as isize;
        match mode {
            Mode::Constant => cval,
            Mode::Symmetric => {
                if (-idx) < len_x {
                    x[(-idx - 1) as usize]
                } else {
                    let idx = (-idx - 1) % (2 * len_x);
                    if idx < len_x {
                        x[idx as usize]
                    } else {
                        x[(len_x - 1 - (idx - len_x)) as usize]
                    }
                }
            }
            Mode::ConstantEdge => x[0],
            Mode::Smooth => *x.first().unwrap() + (x[1] * *x.first().unwrap()) * idx.into(),
            Mode::Periodic => {
                let idx = (-idx - 1) % len_x;
                x[(len_x - idx - 1) as usize]
            }
            Mode::Reflect => {
                if -idx < len_x - 1 {
                    x[(-idx) as usize]
                } else {
                    let idx = (-idx - 1) % (2 * (len_x - 1));
                    if idx < len_x - 1 {
                        x[(idx + 1) as usize]
                    } else {
                        x[(len_x - 2 - (idx - (len_x - 1))) as usize]
                    }
                }
            }
            Mode::Antisymmetric => {
                if -idx < len_x {
                    -x[(-idx - 1) as usize]
                } else {
                    let idx = (-idx - 1) % (2 * (len_x - 1));
                    if idx < len_x - 1 {
                        -x[idx as usize]
                    } else {
                        x[(len_x - 2 - (idx - (len_x - 1))) as usize]
                    }
                }
            }
            Mode::Antireflect => {
                if -idx < len_x {
                    x[0] - (x[(-idx) as usize] - x[0])
                } else {
                    let le = x[0]
                        + (x[0] - x[(len_x - 1) as usize])
                            * (isize::into(-(idx) - 1) / isize::into(len_x - 1));
                    let idx = (-idx - 1) % (2 * (len_x - 1));
                    if idx < (len_x - 1) {
                        le - (x[(idx + 1) as usize] - x[0])
                    } else {
                        let tmp = (len_x - 2 - (idx - (len_x - 1))) as usize;
                        le - (x[len_x as usize - 1] - x[tmp])
                    }
                }
            }
            Mode::Line => {
                let lin_slope = (x[len_x as usize - 1] - x[0]) / (len_x - 1).into();
                x[0] + idx.into() * lin_slope
            }
        }
    }
}
