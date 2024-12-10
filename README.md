# Flask Capstone

> API untuk model dari machine learning pada project capstone epialert.

## Base URL
https://ml.epialert.my.id

## Endpoints

### Predict Image
- URL
    - `/predict`
- Method
    - POST
- Headers
    - `Content-Type`: `multipart/form-data`
- Request Body
    - `file` as `file`, must be a valid image file
- Response
    ```json
    {
        "predicted_class": "string",
        "confidence": "float",
        "code": "string"
    }
    ```