import React, { useState } from "react"
import Webcam from "react-webcam"
import "./AddFace.scss"
import Button from "../../components/Button/Button"
import WebcamStreamCapture from "../../components/WebcamStreamCapture/WebcamStreamCapture"
import BackButton from "../../components/BackButton/BackButton"

const webcamStyle: React.CSSProperties = {
	textAlign: "center",
	height: "35rem",
	width: "25rem",
	objectFit: "cover",
	borderRadius: "2rem",
}

function AddFace() {
	const [uploadCompleted, setUploadCompleted] = useState<boolean>(false)

	return uploadCompleted ? (
		<UploadCompleted />
	) : (
		<>
			<div className="add_face-container background">
				<BackButton link="/" />
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
					<WebcamStreamCapture style={webcamStyle} />
				</div>
			</div>
		</>
	)
}

function UploadCompleted() {
	return <div className="background"></div>
}

export default AddFace
