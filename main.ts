// import { Vec3f, createVector, Point, subtract, dotProduct } from './vector';

const canvas = <HTMLCanvasElement>document.getElementById('canvas');
let ctx = canvas.getContext('2d');
if (!(ctx = canvas.getContext("2d"))) {
    throw new Error(`2d context not supported or canvas already initialized`);
}
canvas.width  = 1014;
canvas.height = 768;
const img = new Image(canvas.width, canvas.height);
const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
const canvasPitch = imgData.width * 4;

interface Vec3f {
    x: number,
    y: number,
    z: number,
}

type Point = Vec3f;

function createVector(x: number, y: number, z: number): Vec3f {
    return {
        x,
        y,
        z
    }
}

function vecLength({ x, y, z }: Vec3f): number {
    return Math.sqrt(x * x + y * y + z * z);
}

function normalize(v: Vec3f): Vec3f {
    const len = vecLength(v); 
    if(len > 0) {
        const inverseLen = 1 / len;
        v.x *= inverseLen;
        v.y *= inverseLen;
        v.z *= inverseLen;
    }

    return v;
}

function dotProduct(v1: Vec3f, v2: Vec3f): number {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

function subtract(v1: Vec3f, v2: Vec3f): Vec3f {
    return createVector(
        v1.x - v2.x,
        v1.y - v2.y,
        v1.z - v2.z
    )
}

function add(v1: Vec3f, v2: Vec3f): Vec3f {
    return createVector(
        v1.x + v2.x,
        v1.y + v2.y,
        v1.z + v2.z
    )
}

function multiply({ x, y, z }: Vec3f, num: number): Vec3f {
    return createVector(
        x * num,
        y * num,
        z * num
    )
}

function putPixel(x: number, y: number, color: number[]): void {
    x = canvas.width / 2 + x;
    y = canvas.height / 2 - y - 1;

    if(x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) {
        return;
    }

    let offset = 4 * x + canvasPitch * y;
    imgData.data[offset++] = color[0];
    imgData.data[offset++] = color[1];
    imgData.data[offset++] = color[2];
    imgData.data[offset++] = 255;
}

type Color = [number, number, number];

interface Ray { 
    origin: Point;
    direction: Vec3f;
    // tMax: number;
}

interface Sphere {
    center: Vec3f;
    radius: number;
    color: Color;

    // doesRayIntersect: (origin: Vec3f, direction: Vec3f, t0: number) => boolean;
    // castRay(origin: Vec3f, direction: Vec3f, sphere: Sphere): Vec3f;
}

function createSphere(center: Vec3f, radius: number, color: Color): Sphere {
    return {
        center,
        radius, 
        color
    }
}

function canvasToViewport(p2d: [number, number]): Vec3f {
    return {
        x: p2d[0] * viewportSize / canvas.width,
        y: p2d[1] * viewportSize / canvas.height,
        z: projectionPlaneZ,
    }; 
}


const camera = {x: 0, y: 0, z: 0};
const objects = [];
const backgroundColor = [255, 255, 255];
const viewportSize = 1;
const projectionPlaneZ = 1;

const spheres = [
    createSphere(
        createVector(0, -1, 3),
        1,
        [255, 0, 0]
    ),
    createSphere(
        createVector(2, 0, 4),
        1,
        [0, 0, 255]
    ),
    createSphere(
        createVector(-2, 0, 4),
        1,
        [0, 225, 0]
    ),
]

function rayIntersection(sphere: Sphere, ray: Ray): [number, number] {
    // const sphereToRay = subtract(ray.origin, sphere.center);
    // const b = 2 * dotProduct(ray.direction, sphereToRay);
    // const c = dotProduct(sphereToRay, sphereToRay) - sphere.radius * sphere.radius;
    // const discriminant = b * b - 4 * c;
    //
    // const dist = -b - Math.sqrt(discriminant) / 2;

    const rayToSphere = subtract(ray.origin, sphere.center);

    const a = dotProduct(ray.direction, ray.direction);
    const b = 2 * dotProduct(rayToSphere, ray.direction);
    const c = dotProduct(rayToSphere, rayToSphere) - sphere.radius * sphere.radius;
    const discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        return [Infinity, Infinity];
    }
    
    // Intersection Points
    const t1 = -b + Math.sqrt(discriminant) / 2;
    const t2 = -b - Math.sqrt(discriminant) / 2;
    return [t1, t2];
}

function castRay(ray: Ray, min_t: number, max_t: number): number[] {
    let closest_t = Infinity;
    let closest_sphere;
    
    for(let i = 0; i < spheres.length; i++) {
        const ts = rayIntersection(spheres[i], ray);
        if(ts[0] < closest_t && min_t < ts[0] && ts[0] < max_t) {
            closest_t = ts[1];
            closest_sphere = spheres[i];
        }
            if (ts[1] < closest_t && min_t < ts[1] && ts[1] < max_t) {
            closest_t = ts[1];
            closest_sphere = spheres[i];
        }
    }

    if(closest_sphere === undefined) {
        return backgroundColor;
    }

    return closest_sphere.color;
}

interface Scene {
    camera: Point;
    objects: Sphere[];
    width: number;
    height: number;
}


function createScene(): Scene {
    return {
        camera: {x: 0, y: 0, z: 0},
        objects: [],
        width: canvas.width,
        height: canvas.height
    }
}

function render(img: ImageData): void {
    for(let x = -canvas.width / 2; x < canvas.width / 2; x++) {
        for(let y = -canvas.height / 2; y < canvas.height / 2; y++) {
            let direction = canvasToViewport([x,y]);
            let color = castRay({origin: camera, direction}, 1, Infinity);
            putPixel(x, y, color);
        }
    }
    // let data = img.data;
    // for(let i = 0; i < data.length; i++) {
    //     data[i] = 128;
    //     data[i + 1] = 128;
    //     data[i + 2] = 128;
    // }
    // console.log(img);
    ctx!.putImageData(img, 0, 0);
}

function main(): void {
    render(imgData);
}

main();
