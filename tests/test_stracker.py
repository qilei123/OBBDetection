import cv2
import sys
 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
            tracker_class = cv2.TrackerBoosting_create
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
            tracker_class = cv2.TrackerMIL_create
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
            tracker_class = cv2.TrackerKCF_create
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
            tracker_class = cv2.TrackerTLD_create
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
            tracker_class = cv2.TrackerMedianFlow_create
        # elif tracker_type == 'GOTURN':
        #     tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
            tracker_class = cv2.TrackerMOSSE_create
        elif tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
            tracker_class = cv2.TrackerCSRT_create
 
    # Read video
    video = cv2.VideoCapture('D:\\DJI_0064.MOV')
    # video = cv2.VideoCapture(0) # for using CAM
 
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()

    scale = 0.5

    frame = cv2.resize(frame, (int(1920*scale), int(1080*scale)))
    
    if not ok:
        print ('Cannot read video file')
        sys.exit()
     
    # Define an initial bounding box
    bbox = (4, 10, 30, 40)
 
    # Uncomment the line below to select a different bounding box
    #bbox = cv2.selectROI(frame, False)
 
    # Initialize tracker with first frame and bounding box
    #ok = tracker.init(frame, bbox)
    inited_tracker = False
    tok = False
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        frame = cv2.resize(frame, (int(1920*scale), int(1080*scale)))

        # Start timer
        timer = cv2.getTickCount()
        if inited_tracker:
            # Update tracker
            tok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if tok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            inited_tracker = False
            tracker = tracker_class()
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        key  = cv2.waitKey(20) & 0xFF
        #if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
        #    break

        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            bbox = cv2.selectROI("Tracking", frame, fromCenter=False,
                                showCrosshair=True) 
            print(bbox)
            ok = tracker.init(frame, bbox) 
            inited_tracker = True
        elif key == ord("q"):
            break              
    video.release()
    cv2.destroyAllWindows()