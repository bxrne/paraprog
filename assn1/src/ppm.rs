use std::fs::File;
use std::io::{self, Write};

/// Write raw RGB pixel data to a binary (P6) PPM file.
pub fn write(path: &str, width: usize, height: usize, pixels: &[u8]) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(pixels)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::remove_file;

    #[test]
    fn test_write_ppm() {
        let width = 2;
        let height = 2;
        let pixels = vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            255, 255, 0, // Yellow
        ];

        let path = "test.ppm";
        write(path, width, height, &pixels).expect("Failed to write PPM file");

        // Read the file back and check its contents
        let contents = std::fs::read(path).expect("Failed to read PPM file");
        let expected_header = b"P6\n2 2\n255\n";
        assert_eq!(&contents[..expected_header.len()], expected_header);
        assert_eq!(&contents[expected_header.len()..], &pixels[..]);

        // Clean up the test file (if ecxists)
        remove_file(path).expect("Failed to remove test PPM file");
    }
}
