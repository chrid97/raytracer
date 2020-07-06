export interface Vec3 {
    x: number,
    y: number,
    z: number,
}

export type Point = Vec3;

export function createVector(x: number, y: number, z: number): Vec3 {
    return {
        x,
        y,
        z
    }
}

export function vecLength(v: Vec3): number {
    return Math.sqrt(dotProduct(v, v));
}


export function normalize(v: Vec3): Vec3 {
    const len = vecLength(v); 
    if(len > 0) {
        const inverseLen = 1 / len;
        v.x *= inverseLen;
        v.y *= inverseLen;
        v.z *= inverseLen;
    }

    return v;
}

export function dotProduct(v1: Vec3, v2: Vec3): number {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

export function subtract(v1: Vec3, v2: Vec3): Vec3 {
    return createVector(
        v1.x - v2.x,
        v1.y - v2.y,
        v1.z - v2.z
    )
}

export function add(v1: Vec3, v2: Vec3): Vec3 {
    return createVector(
        v1.x + v2.x,
        v1.y + v2.y,
        v1.z + v2.z
    )
}

export function multiply({ x, y, z }: Vec3, num: number): Vec3 {
    return createVector(
        x * num,
        y * num,
        z * num
    )
}
