import React from "react"
import Webcam from "react-webcam"
import "./AddFace.scss"
import Button from "../../components/Button/Button"
import WebcamStreamCapture from "../../components/WebcamStreamCapture/WebcamStreamCapture"

const webcamStyle: React.CSSProperties = {
	textAlign: "center",
	height: "35rem",
	width: "25rem",
	objectFit: "cover",
	borderRadius: "2rem",
}

function AddFace() {
	return (
		<div className="add_face-container background">
			<div className="add_face-container__left">
				<p>
					<b>Make sure that</b>
				</p>
				<p>The face is at the center of the frame</p>
				<p>There are no objects or hair covering the face</p>
				<p>The whole face is visible inside of the frame</p>
				<p>The environment is well lit</p>
				<p>The face is angled towards the camera</p>
			</div>

			<div className="add_face-container__right">
				{/* <Webcam audio={false} screenshotFormat="image/jpeg" style={webcamStyle} /> */}
				<WebcamStreamCapture startRecordingText="Start video capturing" endRecordingText="End video capturing" style={webcamStyle} />
			</div>
		</div>
	)
}

export default AddFace
