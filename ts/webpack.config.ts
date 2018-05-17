import * as webpack from 'webpack';

module.exports = {
    entry: './src/main.ts',
    mode: 'development',
    plugins: [new webpack.HotModuleReplacementPlugin()],

    module: {
        rules: [
            {
                test: /\.ts$/,
                use: 'ts-loader',
                exclude: /node_modules/
            }
        ]
    },

    resolve: {
        extensions: ['.ts', '.js']
    },

    output: {
        filename: 'bundle.js',
        path: __dirname
    },

    devServer: {
        hot: true,
        port: 8080
    },

    devtool: 'inline-source-map'
};
