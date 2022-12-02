import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ReactSession } from 'react-client-session';
import axios from 'axios';

import "./Homepage.scss";
import images from "../../constants/images";
import { LogoutButton } from '../../components';
import ProfileIconName from "../../components/ProfileIconName/ProfileIconName";
import Button from "../../components/Button/Button";

type AttendanceRowProps = {
	date: Date
	paid: number
}

function Homepage() {
	const [userPhoto, setUserPhoto] = useState([]);
	const [attendanceList, setAttendanceList] = useState<AttendanceRowProps[]>([]);
	const navigate = useNavigate();

	const firstLetterUppercase = (str: string) => {
		return str.charAt(0).toUpperCase() + str.slice(1);
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
			axios.get('http://localhost:8000/api/get_attendance_list', { params: { id: ReactSession.get('USER_ID') } })
			.then(function(response) {
				setAttendanceList(JSON.parse(response.data.data));
			})
		}
	}, [])

	return (
		<div className="background">
			<ProfileIconName name={ReactSession.get("USER_EMAIL") !== undefined ? firstLetterUppercase(ReactSession.get("USER_NAME"))+" "+firstLetterUppercase(ReactSession.get("USER_SURNAME")) : ''} />
			<LogoutButton />
			{ReactSession.get("USER_ROLE") === 'admin' ? (
				// to implement
				''
			) : (
				<div className="centralContainer">{userPhoto.length > 0 ? <Home attendanceList = {attendanceList} /> : <UploadPhoto />}</div>
			)}
		</div>
	)
}

function Home(props: any) {
	const navigate = useNavigate();
	
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
					{props.attendanceList.map((item: AttendanceRowProps, index: number) =>
						<div key={'attendance'+index}>
							<AttendanceRow date={new Date(item.date)} paid={item.paid}/>
						</div>
					)}
				</div>
			</div>
		</div>
	)
}
function AttendanceRow({ date, paid }: AttendanceRowProps) {
	return (
		<>
			<div className="row">
				<p>{`${date.getDay()}-${date.getMonth()}-${date.getFullYear()}`}</p>
				<p>{`${date.getHours()}:${date.getMinutes()}`}</p>
				<p>{paid % 1 !== 0 ? paid : `${paid}.00`}</p>
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
