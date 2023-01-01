import { Button } from "@mui/material";
import React, { useState } from "react";
import { createRoot } from 'react-dom/client';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import init, { load_and_run } from "../pkg/altius_wasm.js";

const App: React.FC = () => {
  const [image, setImage] = useState<Uint8Array>();
  const [model, setModel] = useState<Uint8Array>();
  const [imgBtnVariant, setImgBtnVariant] = useState('outlined');
    
  const setImgUploaded = () => {
    if (imgBtnVariant === 'outlined') {
      setImgBtnVariant('contained');
    } else {
      setImgBtnVariant('outlined');
    }
  }

  init();

  function loadImage() {
    const img = document.querySelector<HTMLInputElement>("#img");
    if (!img || !img.files) return;

    const file = img.files[0];
    const reader = new FileReader();

    reader.onload = (_: Event) => {
      const loadedImage = new Uint8Array(reader.result as ArrayBufferLike);
      setImage(loadedImage);
      console.log(reader.result as string);
      var blob = new Blob([loadedImage]);
      var urlCreator = window.URL || window.webkitURL;
      var imageUrl = urlCreator.createObjectURL( blob );
      (document.getElementById("image") as HTMLImageElement).src = imageUrl;
    };

    reader.readAsArrayBuffer(file);
    setImgUploaded();
  }

  function loadModel() {
    const onnx = document.querySelector<HTMLInputElement>("#onnx");
    if (!onnx || !onnx.files) return;

    const file = onnx.files[0];
    const reader = new FileReader();

    reader.onload = (_: Event) => {
      const model = new Uint8Array(reader.result as ArrayBufferLike);
      setModel(model);
    };

    reader.readAsArrayBuffer(file);
  }

  function setResultHtml(html: string) {
    (document.getElementById("results") as HTMLDivElement).innerHTML = html;
  }

  function runInference() {
    if (!model || !image) return;
    const msg = load_and_run(model, image);
    setResultHtml(msg);
  }

  return (
    <div id="main">
      <h1 id="title">
        Altius on Web
      </h1>
      <Card sx={{ maxWidth: 400 }} >
        <CardContent>
          <Box 
            display="flex" 
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
          >
            <Box>
              <img id="image" src=""></img>
            </Box>
            <Box>
              <Typography id="results">
                Select an image and an ONNX model first.
              </Typography>
            </Box>
          </Box>
        </CardContent>
        <CardActions>
            <Button size="small" component="label" disabled={image != undefined}>
              Upload image
              <input id="img" name="img" type="file" onChange={loadImage} hidden />
            </Button>
            <Button size="small" component="label" disabled={model != undefined}>
              Upload ONNX Model
              <input id="onnx" name="onnx" type="file" onChange={loadModel} hidden />
            </Button>
            <Button style={{marginLeft:"auto"}} size="small" onClick={runInference} disabled={!image || !model}>
              Run
            </Button>
        </CardActions>
      </Card>
    </div>
  );
};

const container = document.getElementById('app');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}
