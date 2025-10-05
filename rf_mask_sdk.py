from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ngRIEI4jbSnGhy3wxLaL"
)

result = CLIENT.infer("1.jpg", model_id="wall-ceiling-instance-segmentn/1")