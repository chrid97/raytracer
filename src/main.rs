use std::{
    fs::File,
    io::Write,
    ops::{Deref, DerefMut, Index, IndexMut},
    usize,
};

#[derive(Debug)]
struct Tuple([f64; 4]);
impl Tuple {
    fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self([a, b, c, d])
    }
}

impl PartialEq for Tuple {
    fn eq(&self, t: &Tuple) -> bool {
        if equal(self[0], t[0])
            && equal(self[1], t[1])
            && equal(self[2], t[2])
            && equal(self[3], t[3])
        {
            return true;
        }

        false
    }
}

#[derive(Debug, Clone)]
struct Matrix<const N: usize, const M: usize>([[f64; M]; N]);
impl<const N: usize, const M: usize> From<[[f64; M]; N]> for Matrix<N, M> {
    fn from(data: [[f64; M]; N]) -> Self {
        Self(data)
    }
}

impl<const N: usize, const M: usize> Matrix<N, M> {
    // currently only works for 4 by 4 matrix, figure out how to do this for any number
    // although apparently for this book we'll only multiply 4x4 matrices
    // it also says that multiplying the rows is the same as applying dot product to vectors
    // so maybe i can do something with that
    fn mul(lhs: Self, rhs: Self) -> Self {
        let mut matrix = Matrix::from([[0.; M]; N]);
        for row in 0..N {
            for col in 0..M {
                matrix[row][col] = lhs[row][0] * rhs[0][col]
                    + lhs[row][1] * rhs[1][col]
                    + lhs[row][2] * rhs[2][col]
                    + lhs[row][3] * rhs[3][col];
            }
        }

        matrix
    }

    // mul 4x4 matrix by a 4-tuple
    fn mul_tuple(m: Matrix<4, 4>, rhs: Tuple) -> Tuple {
        let mut t = Tuple::new(0., 0., 0., 0.);
        for row in 0..4 {
            for col in 0..4 {
                t[row] += m[row][col] * rhs[col];
            }
        }

        t
    }

    fn identity() -> Matrix<4, 4> {
        Matrix::<4, 4>([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
    }

    fn transpose(self) -> Matrix<M, N> {
        let mut res = [[0.; N]; M];
        for row in 0..N {
            for column in 0..M {
                res[row][column] = self[column][row];
            }
        }

        Matrix(res)
    }
}

impl Index<usize> for Tuple {
    type Output = f64;
    fn index(&self, idx: usize) -> &f64 {
        &self.0[idx]
    }
}

impl IndexMut<usize> for Tuple {
    fn index_mut(&mut self, idx: usize) -> &mut f64 {
        &mut self.0[idx]
    }
}

impl<const N: usize, const M: usize> Deref for Matrix<N, M> {
    type Target = [[f64; M]; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize, const M: usize> DerefMut for Matrix<N, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// review this later
impl<const N: usize, const M: usize> PartialEq for Matrix<N, M> {
    fn eq(&self, other: &Self) -> bool {
        self.0
            .iter()
            .flat_map(|x| x)
            .zip(other.0.iter().flat_map(|y| y))
            .all(|(x, y)| equal(x.clone(), y.clone()))
    }
}

struct Canvas {
    width: usize,
    height: usize,
    pixels: Vec<Color>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![Color::new(0., 0., 0.); width * height],
        }
    }

    fn write_pixel(&mut self, x: usize, y: usize, color: Color) {
        let pos = self.width * y + x;
        self.pixels[pos] = color;
    }

    fn pixel_at(&mut self, x: usize, y: usize) -> Color {
        let pos = self.width * y + x;
        self.pixels[pos]
    }

    fn save_ppm(self) {
        let mut file = File::create("image.ppm").expect("File not found");
        let mut contents = format!("P3\n{}, {}\n255\n", self.width, self.height);
        for pixel in self.pixels {
            contents = contents
                + &(pixel.x as u8).to_string()
                + " "
                + &pixel.y.to_string()
                + " "
                + &pixel.z.to_string()
                + "\n";
        }
        // println!("{}", contents);
        file.write_all(contents.as_bytes())
            .expect("Failed to write to file");
    }
}

#[derive(Debug, Copy, Clone)]
struct Vector {
    x: f64,
    y: f64,
    z: f64,
}

type Point = Vector;
type Color = Vector;

impl Color {}

impl Vector {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vector { x, y, z }
    }

    fn add(Self { x, y, z }: Self, vec2: Vector) -> Self {
        Vector {
            x: x + vec2.x,
            y: y + vec2.y,
            z: z + vec2.z,
        }
    }

    fn sub(Self { x, y, z }: Self, vec2: Vector) -> Self {
        Vector {
            x: x - vec2.x,
            y: y - vec2.y,
            z: z - vec2.z,
        }
    }

    fn negate(self) -> Self {
        Vector {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    fn scalar_mult(self, other: f64) -> Vector {
        Self {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }

    fn divide(self, other: f64) -> Vector {
        Self {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }

    fn magnitude(self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    fn normalize(self) -> Self {
        let magnitude = self.magnitude();
        Vector {
            x: self.x / magnitude,
            y: self.y / magnitude,
            z: self.z / magnitude,
        }
    }

    fn dot(Self { x, y, z }: Self, other: Vector) -> f64 {
        x * other.x + y * other.y + z * other.z
    }

    fn cross(Self { x, y, z }: Self, other: Self) -> Self {
        Self {
            x: y * other.z - z * other.y,
            y: z * other.x - x * other.z,
            z: x * other.y - y * other.x,
        }
    }
}

impl PartialEq for Vector {
    fn eq(&self, vec2: &Vector) -> bool {
        if equal(self.x, vec2.x) && equal(self.y, vec2.y) && equal(self.z, vec2.z) {
            return true;
        }

        false
    }
}

fn equal(a: f64, b: f64) -> bool {
    if (a - b).abs() < 0.00001 {
        return true;
    }

    false
}

struct Projectile {
    position: Point,
    velocity: Vector,
}

struct Environment {
    gravity: Vector,
    wind: Vector,
}

fn main() {
    let mut projectile = Projectile {
        position: Vector::new(0., 1., 0.),
        velocity: Vector::new(1., 1.8, 0.).normalize().scalar_mult(11.25),
    };

    let environment = Environment {
        gravity: Vector::new(0., -0.1, 0.),
        wind: Vector::new(-0.01, 0., 0.),
    };

    let mut canvas = Canvas::new(900, 550);
    let color = Color::new(255., 0., 0.);
    for _ in 0..canvas.width {
        projectile = tick(projectile, &environment);
        if canvas.width * (canvas.height - projectile.position.y as usize)
            + (projectile.position.x as usize)
            < canvas.pixels.len()
        {
            canvas.write_pixel(
                projectile.position.x as usize,
                canvas.height - (projectile.position.y as usize),
                color,
            );
        }
    }

    canvas.save_ppm();
}

fn tick(projectile: Projectile, environment: &Environment) -> Projectile {
    let position = Vector::add(projectile.position, projectile.velocity);
    // find out how to add infinite arguments to the add function
    let velocity = Vector::add(
        Vector::add(projectile.velocity, environment.gravity),
        environment.wind,
    );

    Projectile { position, velocity }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let vec1 = Vector::new(1.1, 2.2, 3.3);
        let vec2 = Vector::new(1.0, 5.0, 10.0);
        let result = Vector::add(vec1, vec2);
        let expected = Vector::new(2.1, 7.2, 13.3);
        assert_eq!(result, expected);
    }

    #[test]
    fn sub() {
        let vec1 = Vector::new(3., 2., 1.);
        let vec2 = Vector::new(5., 6., 7.);
        let result = Vector::sub(vec1, vec2);
        let expected = Vector::new(-2., -4., -6.);
        assert_eq!(result, expected);
    }

    #[test]
    fn negate() {
        let vec = Vector::new(1., -2., 3.);
        let expected = Vector::new(-1., 2., -3.);
        assert_eq!(vec.negate(), expected);
    }

    #[test]
    fn scalar_mult() {
        let vec = Vector::new(1., -2., 3.);
        let expected = Vector::new(3.5, -7., 10.5);
        let expected2 = Vector::new(0.5, -1., 1.5);
        assert_eq!(vec.scalar_mult(3.5), expected);
        assert_eq!(vec.scalar_mult(0.5), expected2);
    }

    #[test]
    fn divide() {
        let vec = Vector::new(1., -2., 3.);
        let expected = Vector::new(0.5, -1., 1.5);
        assert_eq!(vec.divide(2.), expected);
    }

    #[test]
    fn magnitude() {
        {
            let v = Vector::new(1.0, 0.0, 0.0);
            assert_eq!(v.magnitude(), 1.0);
        }
        {
            let v = Vector::new(0., 1., 0.);
            assert_eq!(v.magnitude(), 1.0);
        }
        {
            let v = Vector::new(0.0, 0.0, 1.0);
            assert_eq!(v.magnitude(), 1.0);
        }
        {
            let v = Vector::new(1., 2.0, 3.0);
            assert_eq!(v.magnitude(), f64::sqrt(14.0));
        }
        {
            let v = Vector::new(-1.0, -2.0, -3.0);
            assert_eq!(v.magnitude(), f64::sqrt(14.0));
        }
    }

    #[test]
    fn normalize() {
        {
            let v = Vector::new(4., 0., 0.);
            let expected = Vector::new(1., 0., 0.);
            assert_eq!(v.normalize(), expected);
        }
        {
            let v = Vector::new(1., 2., 3.);
            let expected = Vector::new(0.26726, 0.53452, 0.80178);
            assert_eq!(v.normalize(), expected);
        }
        {
            let v = Vector::new(1., 2., 3.);
            assert_eq!(v.normalize().magnitude(), 1.);
        }
    }

    #[test]
    fn dot() {
        let a = Vector::new(1., 2., 3.);
        let b = Vector::new(2., 3., 4.);
        let product = Vector::dot(a, b);
        assert_eq!(product, 20.)
    }

    #[test]
    fn cross() {
        let a = Vector::new(1., 2., 3.);
        let b = Vector::new(2., 3., 4.);
        let cross_a = Vector::cross(a, b);
        let cross_b = Vector::cross(b, a);
        let expectecd_a = Vector::new(-1., 2., -1.);
        let expectecd_b = Vector::new(1., -2., 1.);
        assert_eq!(cross_a, expectecd_a);
        assert_eq!(cross_b, expectecd_b);
    }

    #[test]
    fn canvas() {
        let c = Canvas::new(10, 20);
        assert_eq!(c.width, 10);
        assert_eq!(c.height, 20);
        for pixel in c.pixels {
            assert_eq!(pixel, Color::new(0., 0., 0.));
        }
        {
            let mut c = Canvas::new(10, 20);
            let red = Color::new(1., 0., 0.);
            c.write_pixel(2, 3, red);
            assert_eq!(c.pixel_at(2, 3), red);
        }
    }

    #[test]
    fn matrix() {
        let m = Matrix::from([
            [1., 2., 3., 4.],
            [5.5, 6.5, 7.5, 8.5],
            [9., 10., 11., 12.],
            [13.5, 14.5, 15.5, 16.5],
        ]);
        let m2 = Matrix::from([
            [1., 2., 3., 4.],
            [5.5, 6.5, 7.5, 8.5],
            [9., 10., 11., 12.],
            [13.5, 14.5, 15.5, 16.5],
        ]);
        let m3 = Matrix::from([
            [20., 22., 50., 48.],
            [44., 54., 114., 108.],
            [40., 58., 110., 102.],
            [16., 26., 46., 42.],
        ]);

        assert_eq!(m[0][0], 1.);
        assert_eq!(m[0][3], 4.);
        assert_eq!(m[1][0], 5.5);
        assert_eq!(m[1][2], 7.5);
        assert_eq!(m[2][2], 11.);
        assert_eq!(m[3][0], 13.5);
        assert_eq!(m[3][2], 15.5);

        assert_eq!(m, m2);
        assert!(m != m3);
    }

    #[test]
    fn matrix_multiplication() {
        let m1 = Matrix::from([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 8., 7., 6.],
            [5., 4., 3., 2.],
        ]);
        let m2 = Matrix::from([
            [-2., 1., 2., 3.],
            [3., 2., 1., -1.],
            [4., 3., 6., 5.],
            [1., 2., 7., 8.],
        ]);
        let expected = Matrix::from([
            [20., 22., 50., 48.],
            [44., 54., 114., 108.],
            [40., 58., 110., 102.],
            [16., 26., 46., 42.],
        ]);

        assert_eq!(Matrix::mul(m1, m2), expected);

        let m = Matrix::from([
            [1., 2., 3., 4.],
            [2., 4., 4., 2.],
            [8., 6., 4., 1.],
            [0., 0., 0., 1.],
        ]);
        let t = Tuple::new(1., 2., 3., 1.);
        let t_expected = Tuple::new(18., 24., 33., 1.);
        assert_eq!(Matrix::<4, 4>::mul_tuple(m, t), t_expected);

        let a = Matrix::<4, 4>::from([
            [0., 1., 2., 4.],
            [1., 2., 4., 8.],
            [2., 4., 8., 16.],
            [4., 8., 16., 32.],
        ]);

        // why do i need to give my identity matrix a type annotation
        // the compiler says it can't infer it but identity() return a 4x4 matrix??
        // so whats there to infer it should already have a type annotation
        assert_eq!(
            a,
            Matrix::<4, 4>::mul(Matrix::<4, 4>::identity(), a.clone())
        );
        assert_eq!(
            Matrix::<4, 4>::mul_tuple(Matrix::<4, 4>::identity(), Tuple::new(1., 2., 3., 4.)),
            Tuple::new(1., 2., 3., 4.)
        );

        let a = Matrix::<4, 4>::from([
            [0., 9., 3., 0.],
            [9., 8., 0., 8.],
            [1., 8., 5., 3.],
            [0., 0., 5., 8.],
        ]);
        let expected = Matrix::<4, 4>::from([
            [0., 9., 1., 0.],
            [9., 8., 8., 0.],
            [3., 0., 5., 5.],
            [0., 8., 3., 8.],
        ]);
        assert_eq!(a.transpose(), expected);
    }
}
