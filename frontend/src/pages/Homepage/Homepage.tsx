import React from "react"
import { useState } from "react"
import "./Homepage.scss"
import images from "../../constants/images"
import ProfileIconName from "../../components/ProfileIconName/ProfileIconName"
import Button from "../../components/Button/Button"

function Homepage() {
	const [hasPhotos, setHasPhotos] = useState(false)

	return (
		<div className="background">
			<ProfileIconName />
			<div className="centralContainer">{hasPhotos ? <Home /> : <UploadPhoto />}</div>
		</div>
	)
}

function Home() {
	// TODO: This has to be still written, it's the component that shows up only
	// if the user has some photos uploaded in the database
	return (
		<>
			<p>Daje</p>
		</>
	)
}

function UploadPhoto() {
	return (
		<>
			<p>You haven't uploaded any photo yet, upload it in order to start using the recognition system!</p>
			<Button text={`Upload a photo of \n your face!`} img={images.face_emoji} />
		</>
	)
}

export default Homepage
