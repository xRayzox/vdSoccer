import streamlit as st
from matplotlib import pyplot as plt
from matplotlib import animation, rc
from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedDistance_Estimator

# Animation function
rc('animation', html='jshtml')

def create_animation(ims):
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    im = plt.imshow(ims[0])

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000 // 24)

# Streamlit app
def main():
    st.title("Soccer Video Analysis and Tracking")

    uploaded_video = st.file_uploader("Upload a soccer video", type=["mp4", "avi"])

    if uploaded_video is not None:
        st.video(uploaded_video)
        st.write("Processing video... This may take a while.")

        # Read video frames
        video_frames = read_video('input_vids/08fd33_4.mp4')

        # Initialize tracker and process tracks
        tracker = Tracker('models/best.pt')
        tracks = tracker.get_obj_tracks(video_frames)
        tracker.add_position_to_track(tracks)

        # Estimate camera movement
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames)
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

        # View Transformer
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)

        # Interpolate ball positions
        if "ball" in tracks and len(tracks["ball"]) > 0:
            tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # Add speed and distance to tracks
        speed_distance_estimator = SpeedDistance_Estimator()
        speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        # Assign players to teams
        team_assigner = TeamAssigner()
        if "players" in tracks and len(tracks["players"]) > 0:
          team_assigner.assign(video_frames[0], tracks["players"][0])

          for frame_num, player_track in enumerate(tracks["players"]):
              for player_id, track in player_track.items():
                  team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
                  tracks["players"][frame_num][player_id]["team"] = team
                  tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

        # Assign ball to player
        player_ball_assigner = PlayerBallAssigner()
        team_ball_control = []

        if "ball" in tracks and len(tracks["ball"]) > 0 and "players" in tracks and len(tracks["players"]) > 0:
            for frame_num, player_track in enumerate(tracks["players"]):
                ball_bbox = tracks["ball"][frame_num][1]["bbox"] if 1 in tracks["ball"][frame_num] else None
                
                if ball_bbox:
                    assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)

                    if assigned_player != -1:
                        tracks["players"][frame_num][assigned_player]["has_ball"] = True        
                        team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
                    else:
                        team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)    

        team_ball_control = np.array(team_ball_control)

        # Draw output
        ## Draw object tracks
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

        ## Draw camera movement
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

        ## Draw speed and distance
        speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)


        # Create animation
        st.write("Displaying processed video...")
        animation_plot = create_animation(output_video_frames)
        st.pyplot(animation_plot)

if __name__ == "__main__":
    main()
