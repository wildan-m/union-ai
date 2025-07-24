from union import ImageSpec, Resources
from union.app import App

image = ImageSpec(
    name="streamlit-app",
    packages=["streamlit==1.41.1", "union-runtime>=0.1.11"],
    registry="ghcr.io/unionai-oss",
)

app1 = App(
    name="streamlit-quickstart",
    container_image=image,
    command="streamlit hello --server.port 8080",
    port=8080,
    limits=Resources(cpu="1", mem="1Gi"),
)
