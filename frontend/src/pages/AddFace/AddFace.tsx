import React, { useState, useEffect, createContext } from "react";
import { useNavigate } from "react-router-dom";
import { ReactSession } from 'react-client-session';

import "./AddFace.scss";
import { BackButton, Button, WebcamStreamCapture } from "../../components";

const webcamStyle: React.CSSProperties = {
	textAlign: "center",
	height: "35rem",
	width: "25rem",
	objectFit: "cover",
	borderRadius: "2rem",
}

const AddFaceContext = createContext({ uploadCompleted: false, setUploadCompleted: (value: boolean) => {} })

function AddFace() {
	const [uploadCompleted, setUploadCompleted] = useState<boolean>(false);
	const navigate = useNavigate();

	useEffect (() => {
        ReactSession.setStoreType("sessionStorage");
		if (ReactSession.get("USER_EMAIL") === undefined)
		{
			navigate('/login');
		}
        if (ReactSession.get("USER_ROLE") === "admin")
        {
            navigate('/');
        }
	}, [])
	

	return (
		<AddFaceContext.Provider value={{ uploadCompleted, setUploadCompleted }}>
			{uploadCompleted ? (
				<UploadCompleted setUploadCompleted={setUploadCompleted} />
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
			)}
		</AddFaceContext.Provider>
	)
}

function UploadCompleted({ setUploadCompleted }: { setUploadCompleted: (value: boolean) => void }) {
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
							text={`Upload another video!`}
							shadow={true}
							onClick={() => {
								setUploadCompleted(false)
								navigate("/add-face")
							}}
						/>
					</div>
				</div>
			</div>
		</>
	)
}

export { AddFaceContext }
export default AddFace
