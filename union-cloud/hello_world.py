# Hello World

import union

image_spec = union.ImageSpec(
    # The name of the image. This image will be used by the `say_hello` task.
    name="say-hello-image",

    # Lock file with dependencies to be installed in the image.
    requirements="uv.lock",

    # Build the image using Union's built-in cloud builder (not locally on your machine).
    builder="union",
)


@union.task(container_image=image_spec)
def say_hello(name: str) -> str:
    return f"Hello, {name}!"

@union.workflow
def hello_world_wf(name: str = "world") -> str:
    greeting = say_hello(name=name)
    return greeting