// import * as tfc from '@tensorflow/tfjs-core';

export namespace Mnist {
    export namespace Image {
        // crops white border from given image
        export interface BoundingBox {
            x1: number;
            y1: number;
            x2: number;
            y2: number;
        }
        export const getBoundigBox = (imageData: ImageData): BoundingBox => {
            const pixelIndex = (x: number, y: number) =>
                4 * (y * imageData.width + x);

            let ax = imageData.width - 1;
            let ay = imageData.height - 1;
            let bx = -1;
            let by = -1;

            for (let y = 0; y <= imageData.height; y++) {
                for (let x = 0; x <= imageData.width; x++) {
                    // read R value as the actual pixel value (ignore G,B,A)
                    if (imageData.data[pixelIndex(x, y)] < 255) {
                        ax = Math.min(ax, x);
                        ay = Math.min(ay, y);
                        bx = Math.max(bx, x);
                        by = Math.max(by, y);
                    }
                }
            }

            return {
                x1: ax,
                y1: ay,
                x2: Math.max(bx, 0),
                y2: Math.max(by, 0)
            };
        };

        export const crop = (
            imageData: ImageData,
            bb: BoundingBox
        ): ImageData => {
            const width = bb.x2 - bb.x1;
            const height = bb.y2 - bb.y1;
            let croppedData = new Uint8ClampedArray(width * height * 4);
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const i_old =
                        4 * ((y + bb.y1) * imageData.width + x + bb.x1);
                    const i = 4 * (y * width + x);
                    croppedData[i] = imageData.data[i_old];
                    croppedData[i + 1] = imageData.data[i_old + 1];
                    croppedData[i + 2] = imageData.data[i_old + 2];
                    croppedData[i + 3] = imageData.data[i_old + 3];
                }
            }
            return new ImageData(croppedData, width, height);
        };

        export interface Tuple {
            x: number;
            y: number;
        }
        export const getCenterOfMass = (imageData: ImageData): Tuple => {
            const i = (x: number, y: number) => 4 * (y * imageData.width + x);

            let sumX = 0;
            let sumY = 0;
            let sum = 0;

            for (let y = 0; y < imageData.height; y++) {
                for (let x = 0; x < imageData.width; x++) {
                    const pv = 1 - imageData.data[i(x, y)] / 255;
                    sumX = sumX + pv * x;
                    sumY = sumY + pv * y;
                    sum += pv;
                }
            }
            return {
                x: Math.floor(sumX / sum),
                y: Math.floor(sumY / sum)
            };
        };

        // export const centerToSquare = (imageData: ImageData) => {
        //     let
        // }
    }

    /*
    1) resize to 20x20 preserving aspect ratio
    2) center in 28x28 canvas using center of mass to be in the center of the canvas
    */
    // const preprocessImage = (document: Document, imageData: ImageData) =>
}
