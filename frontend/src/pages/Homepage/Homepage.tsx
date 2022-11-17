import React, { useEffect, useState } from "react";
import { useNavigate } from 'react-router-dom';
import { ReactSession } from 'react-client-session';

import "./Homepage.scss"
import images from "../../constants/images"
import ProfileIconName from "../../components/ProfileIconName/ProfileIconName"
import Button from "../../components/Button/Button"

type AttendanceRowProps = {
	dateTime: Date
	price: number
}

function Homepage() {
	const [hasPhotos, setHasPhotos] = useState(true);
	const navigate = useNavigate();

	const firstLetterUppercase = (str: string) => {
		return str.charAt(0).toUpperCase() + str.slice(1);
	}

	const logout = () => {
		ReactSession.setStoreType("sessionStorage");
		ReactSession.set("USER_EMAIL", "");
		ReactSession.set("USER_NAME", "");
		ReactSession.set("USER_SURNAME", "");
		ReactSession.set("USER_ROLE", "");
		ReactSession.set("USER_ID", "");
		ReactSession.set("USER_COST", "");
		navigate('/login');
	}

	useEffect (
		() => {
			if (!(ReactSession.get("USER_EMAIL") && ReactSession.get("USER_ROLE")))
			{
				navigate('/login');
			}
		}, []
	)

	return (
		<>
		<div className="background">
			<ProfileIconName name={firstLetterUppercase(ReactSession.get("USER_NAME"))+" "+firstLetterUppercase(ReactSession.get("USER_SURNAME"))} />
			{ReactSession.get("USER_EMAIL") === 'admin' ? (
				// to implement
				''
			) : (
				<div className="centralContainer">{hasPhotos ? <Home /> : <UploadPhoto />}</div>
			)}
		</div>
		<text onClick={logout}>Logout</text>
		</>
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
