import { Vec3, createVector, Point, subtract, dotProduct, vecLength, add, multiply } from './vector.js';

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

function putPixel(x: number, y: number, color: Point): void {
    x = canvas.width / 2 + x;
    y = canvas.height / 2 - y - 1;

    if(x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) {
        return;
    }
    // change color to a interface
    let offset = 4 * x + canvasPitch * y;
    imgData.data[offset++] = color.x;
    imgData.data[offset++] = color.y;
    imgData.data[offset++] = color.z;
    imgData.data[offset++] = 255;
}


interface light {
    type: number;
    intensity: number;
    position: Point;
}

function createLightSource(type: number, intensity: number, position: Vec3): light {
    return {
        type,
        intensity,
        position
    }
}

const lights = [
    createLightSource(0, 0.2, {x: 1, y: 1, z: 1}),
    createLightSource(1, 0.6, {x: 2, y: 1, z: 0}),
    createLightSource(2, 0.2, {x: 1, y: 4, z: 4})
];

function createLighting(point: Point, normal: Point): number {
    let intensity = 0;
    let length_n = vecLength(normal);
    
    for(let i = 0; i < lights.length; i++) {
        if(lights[i].type === 0) {
            intensity += lights[i].intensity;
        } else {
            let lightVector = createVector(0, 0, 0);
            if(lights[i].type === 1) {
                lightVector = subtract(lights[i].position, point);
            } else {
                lightVector = lights[i].position;
            }

            const n_dot_l = dotProduct(normal, lightVector);
            if(n_dot_l > 0) {
                intensity += lights[i].intensity * n_dot_l / (length_n * vecLength(lightVector));
            }
        }
    }

    return intensity;
}

type Color = [number, number, number];

interface Ray { 
    origin: Point;
    direction: Vec3;
    // tMax: number;
}

interface Sphere {
    center: Vec3;
    radius: number;
    color: Color;

    // doesRayIntersect: (origin: Vec3, direction: Vec3, t0: number) => boolean;
    // castRay(origin: Vec3, direction: Vec3, sphere: Sphere): Vec3;
}

function createSphere(center: Vec3, radius: number, color: Color): Sphere {
    return {
        center,
        radius, 
        color
    }
}

function canvasToViewport(p2d: [number, number]): Vec3 {
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

function castRay(ray: Ray, min_t: number, max_t: number): Vec3 {
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
        return {
            x: backgroundColor[0],
            y: backgroundColor[1],
            z: backgroundColor[2],
        }
    }

    const point = add(ray.origin, multiply(ray.direction, closest_t));
    let normal = subtract(point, closest_sphere.center);
    normal = multiply(normal, 1.0 / vecLength(normal));

    return multiply({ 
        x: closest_sphere.color[0],
        y: closest_sphere.color[1],
        z: closest_sphere.color[2]
    }, createLighting(point, normal));
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
    ctx!.putImageData(img, 0, 0);
}

function main(): void {
    render(imgData);
}

main();
