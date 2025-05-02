import cv2
import mediapipe as mp

capture = cv2.VideoCapture(0) #webcam
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


def count_fingers(hand_landmarks, frame_width, frame_height):
    count = 0
    lm = hand_landmarks.landmark

    def to_pixel(landmark):
        return int(landmark.x * frame_width), int(landmark.y * frame_height)

    thumb_tip_x = lm[4].x
    thumb_ip_x = lm[2].x
    if thumb_tip_x > thumb_ip_x:
        count += 1

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    for tip, pip in zip(finger_tips, finger_pips):
        if lm[tip].y < lm[pip].y:
            count += 1

    return count



while True:
    ret, frame = capture.read()
    if not ret:
        print("cannot get frame")
        break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    frame_w = 800
    frame_h = 800
    box_size = 300

    box_left = (frame_w - box_size) // 4
    box_top = (frame_h - box_size) // 4
    box_right = box_left + box_size
    box_bottom = box_top + box_size

    cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)



    if result.multi_hand_landmarks:


        for hand_landmarks in result.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            wrist_x = int(wrist.x * frame_w)
            wrist_y = int(wrist.y * frame_h)
            inside_box = (box_left +10 <= wrist_x <= box_right+75) and (box_top +100 <= wrist_y <= box_bottom +250)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            if inside_box:
                count = count_fingers(hand_landmarks, frame.shape[1], frame.shape[0])
                cv2.putText(frame, f'Fingers Detected: {count}', (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255,255),3)
            else:
                cv2.putText(frame, f'Move fingers into box!', (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0,0),3)






    cv2.imshow("Hand Counter", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

