//! Loops of the time-tagger decoding in `brighteyes.ttm._ttp`.
//!
//! Each kernel repeats the arithmetic of the Cython loop it replaces with the
//! same integer widths, so its results match bit for bit.

use numpy::ndarray::{Array2, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyOverflowError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;

/// Failure of a kernel, raised in Python as the exception the Cython loop raised.
#[derive(Debug, PartialEq)]
pub enum KernelError {
    /// `StepPerPixel` is zero.
    ZeroDivision,
    /// A value does not fit the integer type it is stored in.
    Overflow(String),
    /// A column or row index falls outside the output array.
    Index(usize),
}

impl From<KernelError> for PyErr {
    fn from(error: KernelError) -> Self {
        match error {
            KernelError::ZeroDivision => PyZeroDivisionError::new_err("division by zero"),
            KernelError::Overflow(message) => PyOverflowError::new_err(message),
            KernelError::Index(axis) => {
                PyIndexError::new_err(format!("Out of bounds on buffer access (axis {axis})"))
            }
        }
    }
}

/// Decoded events, one row per event, and the number of rows filled.
pub type Events = (Array2<u8>, usize);

/// Arrays of `t_L`, `t_Ch`, `S_L`, `S_Ch` and `index`, and the number of pairs found.
pub type Pairs = ([Vec<i64>; 5], usize);

/// Arrays of pixel, line, frame and corrected pixel, the row index, and the last frame.
pub type Indices = ([Vec<u16>; 4], Vec<i64>, i32);

fn wrap_index(index: i64, size: usize, axis: usize) -> Result<usize, KernelError> {
    // A negative index counts from the end once, as in a Cython memoryview.
    let size = size as i64;
    let wrapped = if index < 0 { index + size } else { index };
    if (0..size).contains(&wrapped) {
        Ok(wrapped as usize)
    } else {
        Err(KernelError::Index(axis))
    }
}

/// Spread the words of a new-protocol stream into one row per event.
///
/// A channel word below `channels` goes to its own column; the dummy, laser,
/// pixel, scan and line words (ids 123 to 127) go to the last five columns,
/// and a line word that does not count as a channel ends the event. Each of
/// the three blocks of `maxword` columns holds the ids, the valid flags and
/// the data, truncated to `u8`.
pub fn decode_events(
    ids: ArrayView1<u16>,
    valids: ArrayView1<u16>,
    data: ArrayView1<u16>,
    channels: i64,
    maxword: u16,
    maxevents: usize,
) -> Result<Events, KernelError> {
    let maxword = i64::from(maxword);
    let mut events = Array2::<u8>::zeros((maxevents, 3 * maxword as usize));
    let (rows, columns) = events.dim();
    let mut row = 0;
    for ((&id, &valid), &value) in ids.iter().zip(valids).zip(data) {
        let is_channel = i64::from(id) < channels;
        let column = match id {
            _ if is_channel => i64::from(id),
            123..=127 => maxword - 128 + i64::from(id),
            _ => continue,
        };
        let row_index = wrap_index(row as i64, rows, 0)?;
        for (block, word) in [id, valid, value].into_iter().enumerate() {
            let column_index = wrap_index(column + maxword * block as i64, columns, 1)?;
            events[[row_index, column_index]] = word as u8;
        }
        if id == 127 && !is_channel {
            row += 1;
        }
    }
    Ok((events, row))
}

/// Pair each channel photon with the next laser pulse.
///
/// A row with the channel flag set records its time, step and index; the next
/// row with the laser flag set closes the pair with its own time and step.
pub fn pair_photons(
    t_l: ArrayView1<u8>,
    t_ch: ArrayView1<u8>,
    valid_ch: ArrayView1<u8>,
    valid_l: ArrayView1<u8>,
    cumulative_step: ArrayView1<i64>,
    index: ArrayView1<i64>,
) -> Pairs {
    let length = t_l.len();
    let mut pairs: [Vec<i64>; 5] = std::array::from_fn(|_| vec![0; length]);
    let (mut last_t1, mut last_step, mut last_index, mut last_ok) = (0, 0, 0, false);
    let mut count = 0;
    for n in 0..length {
        if valid_ch[n] == 1 {
            last_t1 = i64::from(t_ch[n]);
            last_step = cumulative_step[n];
            last_ok = true;
            last_index = index[n];
        }
        if valid_l[n] == 1 && last_ok {
            last_ok = false;
            let pair = [
                i64::from(t_l[n]),
                last_t1,
                cumulative_step[n],
                last_step,
                last_index,
            ];
            for (column, value) in pairs.iter_mut().zip(pair) {
                column[count] = value;
            }
            count += 1;
        }
    }
    (pairs, count)
}

fn to_u16(value: i32) -> Result<u16, KernelError> {
    u16::try_from(value).map_err(|_| {
        KernelError::Overflow(format!("Python integer {value} out of bounds for uint16"))
    })
}

/// Count pixels, lines and frames from the scan, line and pixel enable flags.
///
/// A rising scan edge starts a new frame and a rising line edge a new line;
/// the corrected pixel is the step distance from the start of the line
/// divided by `step_per_pixel`, truncated toward zero.
pub fn count_pixels(
    scan_enable: ArrayView1<u8>,
    line_enable: ArrayView1<u8>,
    pixel_enable: ArrayView1<u8>,
    cumulative_step: ArrayView1<i64>,
    index: ArrayView1<i64>,
    step_new_line: i64,
    step_per_pixel: i64,
) -> Result<Indices, KernelError> {
    let length = scan_enable.len();
    let mut columns: [Vec<u16>; 4] = std::array::from_fn(|_| vec![0; length]);
    let mut step_new_line = step_new_line;
    let (mut frame, mut line, mut pixel) = (0i32, 0i32, 0i32);
    let (mut last_se, mut last_le) = (0, 0);
    for n in 0..length {
        let (se, le, step) = (scan_enable[n], line_enable[n], cumulative_step[n]);
        if se == 1 && last_se == 0 {
            pixel = 0;
            line = 0;
            frame = frame.wrapping_add(1);
            step_new_line = step;
        }
        if le == 1 && last_le == 0 {
            pixel = 0;
            step_new_line = step;
            line = line.wrapping_add(1);
        }
        if pixel_enable[n] == 1 {
            pixel = pixel.wrapping_add(1);
        }
        if step_per_pixel == 0 {
            return Err(KernelError::ZeroDivision);
        }
        let quotient = (step.wrapping_sub(step_new_line) as f64 / step_per_pixel as f64).trunc();
        if !(f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&quotient) {
            return Err(KernelError::Overflow(
                "Python int too large to convert to C int".to_string(),
            ));
        }
        let pixel_corr = quotient as i32;
        last_se = se;
        last_le = le;
        for (column, value) in columns.iter_mut().zip([pixel, pixel_corr, line, frame]) {
            column[n] = to_u16(value)?;
        }
    }
    Ok((columns, index.to_vec(), frame))
}

fn same_length(lengths: &[usize]) -> PyResult<()> {
    if lengths.windows(2).all(|pair| pair[0] == pair[1]) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "arrays differ in length: {lengths:?}"
        )))
    }
}

/// Spread the words of a new-protocol stream into one row per event.
#[pyfunction]
fn new_protocol_events<'py>(
    py: Python<'py>,
    ids: PyReadonlyArray1<'py, u16>,
    valids: PyReadonlyArray1<'py, u16>,
    data: PyReadonlyArray1<'py, u16>,
    channels: i64,
    maxword: u16,
    maxevents: usize,
) -> PyResult<(Bound<'py, PyArray2<u8>>, usize)> {
    same_length(&[ids.len()?, valids.len()?, data.len()?])?;
    let (events, count) = decode_events(
        ids.as_array(),
        valids.as_array(),
        data.as_array(),
        channels,
        maxword,
        maxevents,
    )?;
    Ok((events.into_pyarray(py), count))
}

/// Pair each channel photon with the next laser pulse.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
fn raw_tdc_pairs<'py>(
    py: Python<'py>,
    t_l: PyReadonlyArray1<'py, u8>,
    t_ch: PyReadonlyArray1<'py, u8>,
    step: PyReadonlyArray1<'py, u16>,
    valid_ch: PyReadonlyArray1<'py, u8>,
    valid_l: PyReadonlyArray1<'py, u8>,
    cumulative_step: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    usize,
)> {
    same_length(&[
        t_l.len()?,
        t_ch.len()?,
        step.len()?,
        valid_ch.len()?,
        valid_l.len()?,
        cumulative_step.len()?,
        index.len()?,
    ])?;
    let ([t_l, t_ch, s_l, s_ch, index], count) = pair_photons(
        t_l.as_array(),
        t_ch.as_array(),
        valid_ch.as_array(),
        valid_l.as_array(),
        cumulative_step.as_array(),
        index.as_array(),
    );
    Ok((
        t_l.into_pyarray(py),
        t_ch.into_pyarray(py),
        s_l.into_pyarray(py),
        s_ch.into_pyarray(py),
        index.into_pyarray(py),
        count,
    ))
}

/// Count pixels, lines and frames from the scan, line and pixel enable flags.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
fn image_indices<'py>(
    py: Python<'py>,
    scan_enable: PyReadonlyArray1<'py, u8>,
    line_enable: PyReadonlyArray1<'py, u8>,
    pixel_enable: PyReadonlyArray1<'py, u8>,
    cumulative_step: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    step_new_line: i64,
    step_per_pixel: i64,
) -> PyResult<(
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<i64>>,
    i32,
)> {
    same_length(&[
        scan_enable.len()?,
        line_enable.len()?,
        pixel_enable.len()?,
        cumulative_step.len()?,
        index.len()?,
    ])?;
    let ([pixel, pixel_corr, line, frame], index, last_frame) = count_pixels(
        scan_enable.as_array(),
        line_enable.as_array(),
        pixel_enable.as_array(),
        cumulative_step.as_array(),
        index.as_array(),
        step_new_line,
        step_per_pixel,
    )?;
    Ok((
        pixel.into_pyarray(py),
        line.into_pyarray(py),
        frame.into_pyarray(py),
        pixel_corr.into_pyarray(py),
        index.into_pyarray(py),
        last_frame,
    ))
}

#[pymodule]
fn _ttp_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(new_protocol_events, module)?)?;
    module.add_function(wrap_pyfunction!(raw_tdc_pairs, module)?)?;
    module.add_function(wrap_pyfunction!(image_indices, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{Array1, arr1};

    fn indices(enable: &[u8], steps: &[i64], step_per_pixel: i64) -> Result<Indices, KernelError> {
        let zeros = Array1::<u8>::zeros(enable.len());
        let flags = arr1(enable);
        let steps = arr1(steps);
        let index = Array1::<i64>::zeros(enable.len());
        count_pixels(
            flags.view(),
            zeros.view(),
            flags.view(),
            steps.view(),
            index.view(),
            steps[0],
            step_per_pixel,
        )
    }

    #[test]
    fn zero_step_per_pixel_divides_by_zero() {
        assert_eq!(indices(&[1], &[5], 0), Err(KernelError::ZeroDivision));
    }

    #[test]
    fn negative_corrected_pixel_overflows_uint16() {
        let error = indices(&[1, 0], &[100, 90], 1).unwrap_err();
        assert_eq!(
            error,
            KernelError::Overflow("Python integer -10 out of bounds for uint16".to_string())
        );
    }

    #[test]
    fn corrected_pixel_outside_i32_overflows() {
        let result = indices(&[1, 0], &[0, i64::MAX], 1);
        assert!(matches!(result, Err(KernelError::Overflow(message)) if message.contains("C int")));
    }

    #[test]
    fn step_distance_wraps_and_divides_as_f64() {
        let ([_, truncated, _, _], _, _) = indices(&[1, 0], &[0, 7], 2).unwrap();
        assert_eq!(truncated[1], 3);
        let ([_, wrapped, _, _], _, _) = indices(&[1, 0], &[i64::MAX, i64::MIN], 1).unwrap();
        assert_eq!(wrapped[1], 1);
    }

    #[test]
    fn stored_words_truncate_to_u8() {
        let ids = arr1(&[0u16, 127]);
        let valids = arr1(&[1u16, 1]);
        let data = arr1(&[0x1FFu16, 0x123]);
        let (events, count) =
            decode_events(ids.view(), valids.view(), data.view(), 1, 6, 2).unwrap();
        assert_eq!(count, 1);
        assert_eq!(events[[0, 12]], 0xFF);
        assert_eq!(events[[0, 17]], 0x23);
    }
}
