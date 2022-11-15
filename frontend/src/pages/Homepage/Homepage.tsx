import React from "react"
import { useState } from "react"
import "./Homepage.scss"
import images from "../../constants/images"
import ProfileIconName from "../../components/ProfileIconName/ProfileIconName"
import Button from "../../components/Button/Button"

type AttendanceRowProps = {
	dateTime: Date
	price: number
}

function Homepage() {
	const [hasPhotos, setHasPhotos] = useState(true)

	return (
		<div className="background">
			<ProfileIconName name="Domiziano Scarcelli" />
			<div className="centralContainer">{hasPhotos ? <Home /> : <UploadPhoto />}</div>
		</div>
	)
}

function Home() {
	// TODO: This has to be still written, it's the component that shows up only
	// if the user has some photos uploaded in the database
	return (
		<div className="infoContainer">
			<div className="leftContainer">
				<Button text="Add another photo!" img={images.selfie_emoji} shadow={false} />
				<Button text="Your photos" img={images.many_faces_emoji} shadow={false} />
				<Button text="Your details" img={images.details_emoji} shadow={false} />
			</div>

			<div className="history">
				<div className="top_history">
					<p>Attendance History</p>
					<img alt="history emoji" src={images.history_emoji}></img>
				</div>
				<div className="body_history">
					<div className="row darker">
						<p>Date</p>
						<p>Time</p>
						<p>Paid</p>
					</div>
					<AttendanceRow dateTime={new Date()} price={3.0} />
					<AttendanceRow dateTime={new Date()} price={3.0} />
					<AttendanceRow dateTime={new Date()} price={3.0} />
				</div>
			</div>
		</div>
	)
}
function AttendanceRow({ dateTime, price }: AttendanceRowProps) {
	return (
		<>
			<div className="row">
				<p>{`${dateTime.getDay()}/${dateTime.getMonth()}/${dateTime.getFullYear()}`}</p>
				<p>{`${dateTime.getHours()}:${dateTime.getMinutes()}`}</p>
				<p>{price % 1 !== 0 ? price : `${price}.00`}</p>
			</div>
		</>
	)
}
function UploadPhoto() {
	return (
		<>
			<p>You haven't uploaded any photo yet, upload it in order to start using the recognition system!</p>
			<Button text={`Upload a photo of \n your face!`} img={images.face_emoji} shadow={true} />
		</>
	)
}

export default Homepage
