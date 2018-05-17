import * as tfc from '@tensorflow/tfjs-core';
import {
    loadFrozenModel,
    NamedTensorMap,
    FrozenModel
} from '@tensorflow/tfjs-converter';

const MODEL_URL = '/saved_web/tensorflowjs_model.pb';
const WEIGHTS_URL = '/saved_web/weights_manifest.json';

const IMG_SIZE = 28;
const IMG_SIZE_FLAT = IMG_SIZE * IMG_SIZE;

const get2DContext = (el: HTMLCanvasElement) => el.getContext('2d');
const getImageData = (ctx: CanvasRenderingContext2D) =>
    ctx.getImageData(0, 0, 400, 400);
const getCanvasElement = () =>
    <HTMLCanvasElement>document.getElementById('main');
const getClearElement = () => document.getElementById('clear');

const runPaint = (
    canvasElement: HTMLCanvasElement | null,
    clearElement: HTMLElement | null
) => {
    if (!canvasElement || !clearElement) {
        throw new Error('Some elements are not defined.');
    }

    const ctx = get2DContext(canvasElement);
    let paintStarted = false;

    if (ctx) {
        ctx.lineWidth = 40;
        ctx.strokeStyle = '#000';
        ctx.fillStyle = '#fff';

        ctx.fillRect(0, 0, canvasElement.width, canvasElement.height);

        const mouseDownHandler = (e: MouseEvent) => (paintStarted = true);
        const mouseUpHandler = (e: MouseEvent) => (paintStarted = false);
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
        canvasElement.addEventListener('mouseleave', mouseUpHandler, false);
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
    const resizedImage = tfc.image.resizeBilinear(tfc.fromPixels(imageData), [
        28,
        28
    ]);

    const x = resizedImage
        .slice(0, [28, 28, 1])
        .reshape([1, IMG_SIZE_FLAT])
        .cast('float32')
        .div(tfc.fill([1, IMG_SIZE_FLAT], 255).cast('float32'));

    x.print();

    const input = {
        x: x
    } as NamedTensorMap;
    return (model.execute(input) as tfc.Tensor).flatten();
};

const main = async () => {
    const canvas = getCanvasElement();
    const ctx = get2DContext(canvas);
    if (canvas && ctx) {
        const model = await loadModel();
        window.setInterval(() => {
            const y = predict(model, getImageData(ctx));
            y.argMax().print();
            console.log(`prediction: ${y.argMax()}`);
        }, 2000);
    }
};

main();
runPaint(getCanvasElement(), getClearElement());
