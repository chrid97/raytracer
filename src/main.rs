use std::f64::EPSILON;

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
        fn equal(a: f64, b: f64) -> bool {
            if (a - b).abs() < 0.00001 {
                return true;
            }

            false
        }

        if equal(self.x, vec2.x) && equal(self.y, vec2.y) && equal(self.z, vec2.z) {
            return true;
        }

        false
    }
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
        velocity: Vector::new(1., 1., 0.).normalize(),
    };

    let environment = Environment {
        gravity: Vector::new(0., -0.1, 0.),
        wind: Vector::new(-0.01, 0., 0.),
    };

    while projectile.position.y > 0. {
        projectile = tick(projectile, &environment);
        println!(
            "{}, {}, {}",
            projectile.position.x, projectile.position.y, projectile.position.z
        );
    }
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
        // let vec1 = Vector::new(3., -2., 5.);
        // let vec2 = Vector::new(-2., 3., 1.);
        // let result = Vector::add(vec1, vec2);
        // let expected = Vector::new(1., 1., 6.);
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
}
