import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ReactSession } from 'react-client-session';
import axios from 'axios';

import "./Homepage.scss"
import images from "../../constants/images"
import ProfileIconName from "../../components/ProfileIconName/ProfileIconName"
import Button from "../../components/Button/Button"

type AttendanceRowProps = {
	dateTime: Date
	price: number
}

function Homepage() {
	const [userPhoto, setUserPhoto] = useState([])
	const navigate = useNavigate();

	const firstLetterUppercase = (str: string) => {
		return str.charAt(0).toUpperCase() + str.slice(1);
	}

	const logout = () => {
		ReactSession.set("USER_EMAIL", "");
		ReactSession.set("USER_NAME", "");
		ReactSession.set("USER_SURNAME", "");
		ReactSession.set("USER_ROLE", "");
		ReactSession.set("USER_ID", "");
		ReactSession.set("USER_COST", "");
		navigate('/login');
	}

	useEffect (() => {
		ReactSession.setStoreType("sessionStorage");
		if (ReactSession.get("USER_EMAIL") === undefined)
		{
			navigate('/login');
		} else {
			axios.get('http://localhost:8000/api/get_photo_list', { params: { id: ReactSession.get('USER_ID') } })
			.then(function(response) {
				setUserPhoto(JSON.parse(response.data.data));
			})
		}
	}, [])

	return (
		<>
		<div className="background">
			<ProfileIconName name={ReactSession.get("USER_EMAIL") !== undefined ? firstLetterUppercase(ReactSession.get("USER_NAME"))+" "+firstLetterUppercase(ReactSession.get("USER_SURNAME")) : ''} />
			{ReactSession.get("USER_ROLE") === 'admin' ? (
				// to implement
				''
			) : (
				<div className="centralContainer">{userPhoto.length > 0 ? <Home userPhoto = {userPhoto}/> : <UploadPhoto />}</div>
			)}
		</div>
		<text onClick={logout}>Logout</text>
		</>
	)
}

function Home(props: any) {
	const [attendanceList, setAttendanceList] = useState([]);
	const navigate = useNavigate();

	useEffect (() => {
		axios.get('http://localhost:8000/api/get_attendance_list', { params: { id: ReactSession.get('USER_ID') } })
		.then(function(response) {
			setAttendanceList(JSON.parse(response.data.data));
		})
	}, [])
	
	return (
		<div className="infoContainer">
			<div className="leftContainer">
				<Button
					text="Add another photo!"
					img={images.selfie_emoji}
					shadow={false}
					onClick={() => {
						console.log("Clicked add photo")
					}}
				/>
				<Button text="Your photos" img={images.many_faces_emoji} shadow={false} onClick={() => {navigate('/get-faces', { state: {userPhoto: props.userPhoto }})}}></Button>
				<Button text="Your details" img={images.details_emoji} shadow={false} onClick={() => {}} />
			</div>

			<div className="history">
				<div className="top">
					<p>Attendance History</p>
					<img alt="history emoji" src={images.history_emoji}></img>
				</div>
				<div className="body">
					<div className="row darker">
						<p>Date</p>
						<p>Time</p>
						<p>Paid</p>
					</div>
					{attendanceList.map((item, index) =>
						<>
							{Object(item).date}
							<AttendanceRow dateTime={new Date(Object(item).date)} price={Object(item).paid}/>
						</>
					)}
				</div>
			</div>
		</div>
	)
}
function AttendanceRow({ dateTime, price }: AttendanceRowProps) {
	return (
		<>
			<div className="row">
				<p>{`${dateTime.getDay()}-${dateTime.getMonth()}-${dateTime.getFullYear()}`}</p>
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
			<Button text={`Upload a photo of \n your face!`} img={images.face_emoji} shadow={true} onClick={() => {}} />
		</>
	)
}

export default Homepage
