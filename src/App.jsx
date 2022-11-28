import { useRef, useEffect } from "react";

// importing dependencies mentioned in the tensorflow documentation
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

// importing utilites and other packages
import Webcam from "react-webcam";
import "./App.css";

function App() {
  const webCamRef = useRef(null);
  const canvasRef = useRef(null);

  //calling movenet function as soon as we have webcam and canvas references
  useEffect(() => {
    loadMovenet();
  }, [webCamRef, canvasRef]);

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user",
  };

  // loading movenet lightning pose from tensorflow website
  async function loadMovenet() {
    try {
      // reference to our canvas element
      const ctx = canvasRef.current.getContext("2d");

      const detectorConfig = {
        modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
        enableTracking: true,
        trackerType: poseDetection.TrackerType.BoundingBox,
        multiPoseMaxDimension: 128,
      };

      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        detectorConfig
      );


      if (detector !== null && detector !== undefined) {
        console.log("Movenet Loaded");
      }

      // calling estimate pose function by passing detector and canvas context to it
      estimatePoses(detector, ctx);
    } catch (error) {
      console.log(error);
    }
  }

  // function to estimate poses after movenet is loaded
  async function estimatePoses(detector, ctx) {
    try {
      const poses = await detector.estimatePoses(webCamRef.current.video);
      let keypoints = poses[0].keypoints;
      // console.log(keypoints);
      if (keypoints == undefined) {
        loadMovenet();
      }
      assignPoints(keypoints, ctx);
      drawKeypoints(keypoints, 0.3, ctx);

      setTimeout(() => {
        ctx.clearRect(
          0,
          0,
          webCamRef.current.video.videoWidth,
          webCamRef.current.video.videoHeight
        );
        estimatePoses(detector, ctx);
      }, 800);
    } catch (error) {
      console.log(error);
    }
  }

  // function to find keypoints and draw lines connecting them to form a skeleton
  function assignPoints(keypoints, ctx) {
    const leftsh = []; // 5
    const rightsh = []; // 6
    const leftelbow = []; // 7
    const leftpalm = []; // 9
    const rightelbow = []; // 8
    const rightpalm = []; // 10
    const lefthip = []; // 11
    const righthip = []; // 12
    const leftknee = []; // 13
    const rightknee = []; // 14
    const leftfoot = []; // 15
    const rightfoot = []; // 16

    // assigning co-ordinates to each point for every keypoints iteration
    for (const [i, value] of keypoints.entries()) {
      // destructuring name , score aka confidence , x and y co-ordiantes from the individual keypoint array
      const { name, score, x, y } = value;

      //assigning co-ordinates of left shoulder point
      if (i === 5 && score > 0.3) {
        storeCoordinate(x, y, leftsh);
      }

      //assigning co-ordinates of right shoulder point
      if (i === 6 && score > 0.3) {
        storeCoordinate(x, y, rightsh);
      }

      //assigning co-ordinates of left elbow point
      if (i === 7 && score > 0.3) {
        storeCoordinate(x, y, leftelbow);
      }

      //assigning co-ordinates of left palm point
      if (i === 9 && score > 0.3) {
        storeCoordinate(x, y, leftpalm);
      }

      //assigning co-ordinates of right elbow point
      if (i === 8 && score > 0.3) {
        storeCoordinate(x, y, rightelbow);
      }

      //assigning co-ordinates of right palm point
      if (i === 10 && score > 0.3) {
        storeCoordinate(x, y, rightpalm);
      }

      //assigning co-ordinates of left hip point
      if (i === 11 && score > 0.3) {
        storeCoordinate(x, y, lefthip);
      }

      //assigning co-ordinates of right hip point
      if (i === 12 && score > 0.3) {
        storeCoordinate(x, y, righthip);
      }

      //assigning co-ordinates of left knee point
      if (i === 13 && score > 0.3) {
        storeCoordinate(x, y, leftknee);
      }

      //assigning co-ordinates of right knee point
      if (i === 14 && score > 0.3) {
        storeCoordinate(x, y, rightknee);
      }

      //assigning co-ordinates of left foot point
      if (i === 15 && score > 0.3) {
        storeCoordinate(x, y, leftfoot);
      }

      //assigning co-ordinates of right foot point
      if (i === 16 && score > 0.3) {
        storeCoordinate(x, y, rightfoot);
      }

      //function to assign points
      function storeCoordinate(xVal, yVal, array) {
        array.push({ x: xVal, y: yVal });
      }
    }

    //drawing upper joint
    drawLine(leftsh[0]?.x, leftsh[0]?.y, rightsh[0]?.x, rightsh[0]?.y, ctx);

    //drawing left arm
    drawLine(leftsh[0]?.x, leftsh[0]?.y, leftelbow[0]?.x, leftelbow[0]?.y, ctx);

    //drawing left forearm
    drawLine(
      leftelbow[0]?.x,
      leftelbow[0]?.y,
      leftpalm[0]?.x,
      leftpalm[0]?.y,
      ctx
    );

    //drawing right arm
    drawLine(
      rightsh[0]?.x,
      rightsh[0]?.y,
      rightelbow[0]?.x,
      rightelbow[0]?.y,
      ctx
    );

    //drawing right forearm
    drawLine(
      rightelbow[0]?.x,
      rightelbow[0]?.y,
      rightpalm[0]?.x,
      rightpalm[0]?.y,
      ctx
    );

    //drawing hip connection
    drawLine(lefthip[0]?.x, lefthip[0]?.y, righthip[0]?.x, righthip[0]?.y, ctx);

    //drawing left upper to lower body connection
    drawLine(leftsh[0]?.x, leftsh[0]?.y, lefthip[0]?.x, lefthip[0]?.y, ctx);

    //drawing right upper to lower body connection
    drawLine(rightsh[0]?.x, rightsh[0]?.y, righthip[0]?.x, righthip[0]?.y, ctx);

    //drawing left thigh
    drawLine(lefthip[0]?.x, lefthip[0]?.y, leftknee[0]?.x, leftknee[0]?.y, ctx);

    //drawing right thigh
    drawLine(
      righthip[0]?.x,
      righthip[0]?.y,
      rightknee[0]?.x,
      rightknee[0]?.y,
      ctx
    );

    //drawing left leg
    drawLine(
      leftknee[0]?.x,
      leftknee[0]?.y,
      leftfoot[0]?.x,
      leftfoot[0]?.y,
      ctx
    );

    //drawing right leg
    drawLine(
      rightknee[0]?.x,
      rightknee[0]?.y,
      rightfoot[0]?.x,
      rightfoot[0]?.y,
      ctx
    );
  }

  // function to draw the skeleton lines
  function drawLine(x1, y1, x2, y2, ctx) {
    ctx.save();
    ctx.translate(canvasRef.current.width, 0);
    ctx.scale(-1, 1);
    ctx.beginPath();
    ctx.lineWidth = 8;
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = "#FF0000";
    ctx.stroke();
    ctx.fillStyle = "red";
    ctx.fill();
    ctx.restore();
  }

  // function to find keypoints and draw them
  function drawKeypoints(keypoints, minscore, ctx, scale = 1) {
    // ctx.scale(-1, 1);
    for (const element of keypoints) {
      const keypoint = element;

      if (keypoint.score < minscore) {
        continue;
      }

      const x = keypoint.x;
      const y = keypoint.y;

      drawPoint(ctx, y * scale, x * scale, 3);
    }
  }

  // function to draw a point
  function drawPoint(ctx, y, x, r) {
    ctx.save();
    ctx.translate(canvasRef.current.width, 0);
    ctx.scale(-1, 1);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = "red";
    ctx.fill();
    ctx.restore();
  }

  return (
    <div className="App">
      <Webcam
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          marginLeft: "auto",
          marginRight: "auto",
          textAlign: "center",
          zindex: 9,
          width: 1280,
          height: 720,
        }}
        ref={webCamRef}
        id="video"
        audio={false}
        mirrored={true}
        videoConstraints={videoConstraints}
      />
      <canvas
        id="canvas"
        ref={canvasRef}
        width="1280"
        height="720"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          marginLeft: "auto",
          marginRight: "auto",
          textAlign: "center",
          zindex: 11,
          width: 1280,
          height: 720,
        }}
      ></canvas>
    </div>
  );
}

export default App;
