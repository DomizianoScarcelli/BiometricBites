import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { ReactSession } from "react-client-session";
import Webcam from "react-webcam";
import { CameraOptions, useFaceDetection } from 'react-use-face-detection';
import FaceDetection from '@mediapipe/face_detection';
import { Camera } from '@mediapipe/camera_utils';
import axios from "axios";

import "./AddFace.scss";
import { BackButton, Button } from "../../components";

const webcamStyle: React.CSSProperties = {
	textAlign: "center",
	height: "35rem",
	width: "25rem",
	objectFit: "cover",
	borderRadius: "2rem",
}

function AddFace() {
	const navigate = useNavigate()
	//const webcamRef = useRef<any>()
	const [photoList, setPhotoList] = useState<Array<string>>([])
	const [instruction, setInstruction] = useState<string>("")
	const [isUserDetected, setIsUserDetected] = useState<boolean>(false)

	const { webcamRef, boundingBox, isLoading, detected, facesDetected }: any = useFaceDetection({
		faceDetectionOptions: {
			model: 'short',
		},
		faceDetection: new FaceDetection.FaceDetection({
			locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`,
		}),
		camera: ({ mediaSrc, onFrame, width, height }: CameraOptions) =>
			new Camera(mediaSrc, { onFrame, width, height }),
	});
	
	
	useEffect(() => {
		if (detected) setIsUserDetected(true)
		else setIsUserDetected(false)
	  }, [detected]);

	useEffect(() => {
		const decideIstruction = (): string => {
			switch (photoList.length) {
				case 0:
					return "Take a photo of your face in your neutral expression!"
				case 1:
					return "Take a photo of your face while smiling!"
				case 2:
					return "Take a photo on one of your profiles!"
				case 3:
					return "Take a photo on the opposite profile!"
				default:
					return ""
			}
		}
		setInstruction(decideIstruction)
	}, [photoList.length])

	const takePhoto = () => {
		const screenshot: string = webcamRef.current.getScreenshot()
		setPhotoList([...photoList, screenshot])
		console.log(photoList)
	}

	useEffect(() => {
		ReactSession.setStoreType("sessionStorage")
		if (ReactSession.get("USER_EMAIL") === undefined) {
			navigate("/login")
		}
		if (ReactSession.get("USER_ROLE") === "admin") {
			navigate("/")
		}
	}, [navigate])

	return photoList.length === 4 ? (
		<ConfirmUpload photoList={photoList} setPhotoList={setPhotoList} />
	) : (
		<>
			<div className="add_face-container background">
				<BackButton link="/" />
				<div className="add_face-container__left">
					<p>
						<b>Make sure that:</b>
					</p>
					<p>The face is at the center of the frame</p>
					<p>There are no objects or hair covering the face</p>
					<p>The whole face is visible inside of the frame</p>
					<p>The environment is well lit</p>
					<p className="istruction">{instruction}</p>
					<div className="photos">
						{photoList.map((photo, index) => (
							<img key={index} src={photo} alt="Face" />
						))}
					</div>
				</div>

				<div className="add_face-container__right">
					<>
						<Webcam mirrored audio={false} ref={webcamRef} screenshotFormat="image/jpeg" forceScreenshotSourceSize style={webcamStyle} />
						{isUserDetected ? (<Button text={"Take photo"} onClick={takePhoto} shadow={true} />) : (<Button text={"User not identified!"} onClick={() => {}} shadow={true} />)}
					</>
				</div>
			</div>
		</>
	)
}

type ConfirmUploadProps = {
	photoList: Array<string>
	setPhotoList: (value: Array<string>) => void
}

const ConfirmUpload = ({ photoList, setPhotoList }: ConfirmUploadProps) => {
	const [uploadComplete, setUploadComplete] = useState<boolean>(false)

	const handleUploadPhoto = async () => {
		//Make the api request to upload the photos
		let formData = new FormData()
		formData.append("photoList", JSON.stringify(photoList))
		formData.append("id", ReactSession.get("USER_ID"))
		await axios.post("http://localhost:8000/api/upload_photo_enrollment", formData)
		setUploadComplete(true)
	}

	return uploadComplete ? (
		<UploadCompleted setUploadComplete={setUploadComplete} setPhoto={setPhotoList}/>
	) : (
		<>
			<div className="background center">
				<div className="centralContainer">
					<p>These are your photos</p>
					<div className="photos">
						{photoList.map((photo, index) => (
							<img key={index} src={photo} alt="Face" />
						))}
					</div>
					<p>Do you want to upload them?</p>
					<div className="buttons">
						<Button
							text={`Retake photos`}
							shadow={true}
							onClick={() => {
								setPhotoList([])
							}}
						/>
						<Button
							text={`Upload photos`}
							shadow={true}
							onClick={() => {
								handleUploadPhoto()
							}}
						/>
					</div>
				</div>
			</div>
		</>
	)
}

type UploadCompletedProps = {
	setUploadComplete: (value: boolean) => void
	setPhoto: (value: Array<string>) => void
}

const UploadCompleted = ({ setUploadComplete, setPhoto }: UploadCompletedProps) => {
	const navigate = useNavigate()
	return (
		<>
			<div className="background center">
				<div className="centralContainer">
					<p>Your photo was uploaded correctly!</p>
					<div className="buttons">
						<Button
							text={`Home`}
							shadow={true}
							onClick={() => {
								navigate("/")
							}}
						/>
						<Button
							text={`Upload other photos!`}
							shadow={true}
							onClick={() => {
								setUploadComplete(false)
								setPhoto([])
								navigate("/add-face")
							}}
						/>
					</div>
				</div>
			</div>
		</>
	)
}

export default AddFace
