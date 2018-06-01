import * as tfc from '@tensorflow/tfjs-core';
import {
    loadFrozenModel,
    NamedTensorMap,
    FrozenModel
} from '@tensorflow/tfjs-converter';
import { Mnist } from './Mnist';

const MODEL_URL = '/saved_web/tensorflowjs_model.pb';
const WEIGHTS_URL = '/saved_web/weights_manifest.json';

const IMG_SIZE = 28;
const IMG_SIZE_FLAT = IMG_SIZE * IMG_SIZE;

const get2DContext = (el: HTMLCanvasElement) => el.getContext('2d');
const getImageData = (ctx: CanvasRenderingContext2D) =>
    ctx.getImageData(0, 0, 400, 400);
const getCanvasElementById = (id: string) =>
    <HTMLCanvasElement>document.getElementById(id);
const getPaintCanvasElement = () => getCanvasElementById('paint');
const getPreprocessedCanvasElement = () => getCanvasElementById('preprocessed');
const getClearElement = () => document.getElementById('clear');

const runPaint = (
    canvasElement: HTMLCanvasElement | null,
    clearElement: HTMLElement | null,
    mouseUpCallback: () => void
) => {
    if (!canvasElement || !clearElement) {
        throw new Error('Some elements are not defined.');
    }

    const ctx = get2DContext(canvasElement);
    let paintStarted = false;

    if (ctx) {
        ctx.lineWidth = 20;
        ctx.strokeStyle = '#000';
        ctx.fillStyle = '#fff';
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        ctx.fillRect(0, 0, canvasElement.width, canvasElement.height);

        const mouseDownHandler = (e: MouseEvent) => (paintStarted = true);
        const mouseUpHandler = (e: MouseEvent) => {
            paintStarted = false;
            mouseUpCallback();
        };
        const mouseLeaveHandler = (e: MouseEvent) => {
            paintStarted = false;
        };
        const mouseMoveHandler = (e: MouseEvent) => {
            const x = e.pageX - canvasElement.offsetLeft;
            const y = e.pageY - canvasElement.offsetTop;

            if (!paintStarted) {
                ctx.moveTo(x, y);
                ctx.beginPath();
            } else {
                ctx.lineTo(x, y);
                ctx.stroke();
            }
        };

        canvasElement.addEventListener('mousedown', mouseDownHandler, false);
        canvasElement.addEventListener('mouseup', mouseUpHandler, false);
        canvasElement.addEventListener('mouseleave', mouseLeaveHandler, false);
        canvasElement.addEventListener('mousemove', mouseMoveHandler, false);

        clearElement.addEventListener('click', () =>
            ctx.fillRect(0, 0, canvasElement.width, canvasElement.height)
        );
    }
};

const loadModel = async () => {
    return await loadFrozenModel(MODEL_URL, WEIGHTS_URL);
};

const predict = (model: FrozenModel, imageData: ImageData) => {
    const preprocessedImage = tfc.fromPixels(Mnist.Image.preprocess(imageData));
    tfc.toPixels(preprocessedImage, getPreprocessedCanvasElement());
    const x = preprocessedImage
        .slice(0, [28, 28, 1]) // extracts channel R
        .reshape([1, IMG_SIZE_FLAT])
        .cast('float32')
        .div(tfc.fill([1, IMG_SIZE_FLAT], 255));

    const inverted = tfc.ones([1, IMG_SIZE_FLAT]).sub(x);
    console.log(inverted.dataSync().toString());

    const input = {
        x: inverted
    } as NamedTensorMap;
    return tfc.softmax(model.execute(input) as tfc.Tensor).flatten();
};

const main = async () => {
    const paintCanvas = getPaintCanvasElement();
    const ctx = get2DContext(paintCanvas);
    tfc.setBackend('cpu');
    if (paintCanvas && ctx) {
        const model = await loadModel();

        runPaint(getPaintCanvasElement(), getClearElement(), () => {
            const y = predict(model, getImageData(ctx));
            y.print();
            console.log(`prediction: ${y.argMax()}`);
        });
    }
};

main();
