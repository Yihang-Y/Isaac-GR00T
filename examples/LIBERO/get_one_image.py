# get one image from the dataset, and save it to the local directory
import os
# import 
# use av to read the video and get the first frame
import av

if __name__ == "__main__":
    video_path = "examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/videos/chunk-000/observation.images.image/episode_000000.mp4"
    image_path = "examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/images/episode_000000.jpg"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            image = frame.to_image()
            image.save(image_path)
            break
    print(f"Saved image to {image_path}")