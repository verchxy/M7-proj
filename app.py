from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import io
import os
import tempfile

app = FastAPI(title="Phone Detector API")

custom_classes = ["smartphone", "cell phone", "mobile phone", "iphone"]
model = None


def get_model():
    global model
    if model is None:
        from ultralytics import YOLO
        model = YOLO("yolov8s-world.pt")
        model.set_classes(custom_classes)
    return model


@app.get("/")
def root():
    return {
        "message": "Phone Detector API is running",
        "classes": custom_classes,
        "model_loaded": model is not None
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        detector = get_model()

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            image.save(temp_path)

        results = detector.predict(
            source=temp_path,
            conf=0.25,
            imgsz=640,
            save=False
        )

        detections = []
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names[cls_id]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

        os.remove(temp_path)

        return JSONResponse({
            "success": True,
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        detector = get_model()

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_input:
            input_path = tmp_input.name
            image.save(input_path)

        results = detector.predict(
            source=input_path,
            conf=0.25,
            imgsz=640,
            save=False
        )

        annotated = results[0].plot()

        output_path = input_path.replace(".jpg", "_detected.jpg")
        Image.fromarray(annotated).save(output_path)

        os.remove(input_path)

        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename="detected_result.jpg"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )