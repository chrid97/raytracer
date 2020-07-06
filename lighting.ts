import { Point, Vec3, createVector, dotProduct, vecLength, subtract } from './vector.js';

interface Light {
    type: number;
    intensity: number;
    position: Point;
}

function createLightSource(type: number, intensity: number, position: Vec3): Light {
    return {
        type,
        intensity,
        position
    }
}

function createLighting(point: Point, normal: Point, lights: Light[]): number {
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

export { Light, createLightSource, createLighting };
