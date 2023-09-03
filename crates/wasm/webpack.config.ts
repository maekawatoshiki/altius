import path from 'path';
import { Configuration } from 'webpack';
import WebpackDevServer from 'webpack-dev-server';

const config: Configuration = {
  mode: process.env.WEBPACK_PRODUCTION ? "production" : "development",
  context: path.join(__dirname, 'src'),
  entry: {
    index: [path.resolve(__dirname, "src", "index.tsx")],
  },
  output: {
    path: path.join(__dirname, 'dist'),
    filename: '[name].bundle.js',
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
      },
    ],
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js', '.jsx'],
  },
  optimization: {
    usedExports: !!process.env.WEBPACK_PRODUCTION,
  },
  devtool: process.env.WEBPACK_PRODUCTION ? false : "eval-cheap-source-map",
  devServer: {
    static: {
      directory: "static",
      publicPath: "/",
    },
    onBeforeSetupMiddleware: (devserver: WebpackDevServer) => {
      devserver.app?.use("/", (req, res, next) => {
        console.log(`${req.ip} - ${req.method} - ${req.originalUrl}`);
        next();
      });
    },
  },
};

export default config;
