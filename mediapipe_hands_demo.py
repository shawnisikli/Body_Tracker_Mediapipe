#these are my imports simple enough
import cv2
import mediapipe as mp


#main function
def run_mediapipe_pose():
    
    #this is pulling the necessary modules for the pose detection, drawing is the drawing of the pose and pose is the actual pose detection
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose 
    mp_hand = mp.solutions.hands
    
    #Cap is what is asking to use my webcam and cap.set is what is allowign me to set the width of the frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    
    #the with statement is what i use to get the general pose detection and threshold. The while loop is what is allowing 
    #  me to get the pose detection continuously. 
    with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence = 0.5) as hands, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence = 0.5) as pose:
        while True:
            #the ret is return for the camera and frame is the actual frame being captured
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            # this checks if the return worked or not
            if ret == False:
                break
            
            #here we are feeding in the frame and changing the frame from BGR to RGB because mediapipe requires rgb
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #the results is ensuring that the pose detection is being processed. 
            pose_results = pose.process(image)
            hands_results = hands.process(image)
            #Turn the image back into BGR so that it can be displayed
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #the if statement is checking if the pose landmarks are being detected and if they are then it will draw the pose
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hand.HAND_CONNECTIONS)
            # cv2.imshow is what is being displayed with the drawings of the pose
            cv2.imshow('Mediapipe Pose and Hand', image)
            #this is just giving a key to quit the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    #this is releasing all resources and closing windows to ensure camera is working properly       
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
            run_mediapipe_pose()
            