export namespace Mnist {
    export namespace Image {
        export interface CoordinateTuple {
            x: number;
            y: number;
        }
        export interface SizeTuple {
            width: number;
            height: number;
        }
        export interface BoundingBox {
            x1: number;
            y1: number;
            x2: number;
            y2: number;
        }

        const initImageArrayToWhite = (size: SizeTuple) => {
            const data = new Uint8ClampedArray(size.width * size.height * 4);
            for (let i = 0; i < data.length; i++) {
                data[i] = 255;
            }
            return data;
        };

        /**
         * calculates fitting bounding box for given image on white background
         */
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

        /**
         * crops image to given bounding box
         */
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

        /**
         * calculates center of mass for the given image
         */
        export const getCenterOfMass = (
            imageData: ImageData
        ): CoordinateTuple => {
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

        /**
         * center image to a square image with size given by larger side of input image
         */
        export const centerToSquare = (imageData: ImageData): ImageData => {
            const isLandscape = imageData.width > imageData.height;
            const size = isLandscape ? imageData.width : imageData.height;
            const offsetX = isLandscape
                ? 0
                : Math.floor((size - imageData.width) / 2);
            const offsetY = isLandscape
                ? Math.floor((size - imageData.height) / 2)
                : 0;

            let centeredData = initImageArrayToWhite({
                width: size,
                height: size
            });

            for (let y = 0; y < imageData.height; y++) {
                for (let x = 0; x < imageData.width; x++) {
                    const i = 4 * (y * imageData.width + x);
                    const i_centered = 4 * ((y + offsetY) * size + x + offsetX);

                    centeredData[i_centered] = imageData.data[i];
                    centeredData[i_centered + 1] = imageData.data[i + 1];
                    centeredData[i_centered + 2] = imageData.data[i + 2];
                    centeredData[i_centered + 3] = imageData.data[i + 3];
                }
            }

            return new ImageData(centeredData, size, size);
        };

        /**
         * resizes image using nearest neighbour method
         */
        export const resizeNearestNeighbor = (
            imageData: ImageData,
            newSize: SizeTuple
        ) => {
            const factorX = newSize.width / imageData.width;
            const factorY = newSize.height / imageData.height;
            let resizedData = new Uint8ClampedArray(
                newSize.width * newSize.height * 4
            );

            for (let y = 0; y < newSize.height; y++) {
                for (let x = 0; x < newSize.width; x++) {
                    const i = 4 * (y * newSize.height + x);

                    const x_old = Math.floor(x / factorX);
                    const y_old = Math.floor(y / factorY);
                    const i_old = 4 * (y_old * imageData.height + x_old);

                    resizedData[i] = imageData.data[i_old];
                    resizedData[i + 1] = imageData.data[i_old + 1];
                    resizedData[i + 2] = imageData.data[i_old + 2];
                    resizedData[i + 3] = imageData.data[i_old + 3];
                }
            }

            return new ImageData(resizedData, newSize.width, newSize.height);
        };

        /**
         * preprocesses image using the same rules used to preprocess MNIST images
         *
         * - resize to 20x20 preserving aspect ratio
         * - center in 28x28 canvas using center of mass of 20x20 image
         */
        export const preprocess = (imageData: ImageData) => {
            const DIGITSIZE = 20;
            const ENDSIZE = 28;
            const ppData = initImageArrayToWhite({
                width: ENDSIZE,
                height: ENDSIZE
            });
            const boundingBox = getBoundigBox(imageData);
            const resized = resizeNearestNeighbor(
                centerToSquare(crop(imageData, boundingBox)),
                {
                    width: DIGITSIZE,
                    height: DIGITSIZE
                }
            );
            const com = getCenterOfMass(resized);

            const offsetX = 4 + DIGITSIZE / 2 - com.x;
            const offsetY = 4 + DIGITSIZE / 2 - com.y;

            for (let y = 0; y < resized.height; y++) {
                for (let x = 0; x < resized.width; x++) {
                    const i_resized = 4 * (y * resized.height + x);
                    const i_preprocessed =
                        4 * ((y + offsetY) * ENDSIZE + x + offsetX);
                    ppData[i_preprocessed] = resized.data[i_resized];
                    ppData[i_preprocessed + 1] = resized.data[i_resized + 1];
                    ppData[i_preprocessed + 2] = resized.data[i_resized + 2];
                    ppData[i_preprocessed + 3] = resized.data[i_resized + 3];
                }
            }

            return new ImageData(ppData, ENDSIZE, ENDSIZE);
        };
    }
}
