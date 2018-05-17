import * as tfc from '@tensorflow/tfjs-core';
import { loadFrozenModel, NamedTensorMap } from '@tensorflow/tfjs-converter';

const MODEL_URL = '/saved_web/tensorflowjs_model.pb';
const WEIGHTS_URL = '/saved_web/weights_manifest.json';

const IMG_SIZE = 28;
const IMG_SIZE_FLAT = IMG_SIZE * IMG_SIZE;

const get2DContext = (el: HTMLCanvasElement) => el.getContext('2d');
const getImageData = (ctx: CanvasRenderingContext2D) =>
    ctx.getImageData(0, 0, 28, 28);

const main = async () => {
    // tfc.setBackend('cpu');
    console.log(tfc.getBackend());
    const canvas = <HTMLCanvasElement>document.getElementById('main');
    const ctx = get2DContext(canvas);

    if (canvas && ctx) {
        const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL);

        // const xx = tfc.rand(
        //     [1, IMG_SIZE_FLAT],
        //     () => Math.floor(Math.random() * 256) - 1,
        //     'float32'
        // );
        // xx.print();
        // console.log(xx.shape);

        const r = tfc.fromPixels(getImageData(ctx));

        const input = {
            x: r
                .slice(0, [28, 28, 1])
                .reshape([1, IMG_SIZE_FLAT])
                .cast('float32')
        } as NamedTensorMap;
        const y = model.execute(input) as tfc.Tensor;
        console.log(y);
        y.print();
    }
};

main();
