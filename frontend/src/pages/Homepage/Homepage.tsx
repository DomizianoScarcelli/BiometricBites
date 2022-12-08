import React, { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { ReactSession } from "react-client-session"
import axios from "axios"

import "./Homepage.scss";
import images from "../../constants/images";
import { LogoutButton, ProfileIconName, Button } from "../../components";
import moment, { Moment } from "moment";

import Admin from "../Admin/Admin"

type AttendanceRowProps = {
	date: Moment
	paid: number
}

type AttendanceList = {
	user_id: number
	attendanceId: number
	paid: number
	date: string
}

function Homepage() {
	const [userPhoto, setUserPhoto] = useState<string[]>([]);
	const [attendanceList, setAttendanceList] = useState<AttendanceList[]>([]);
	const navigate = useNavigate();

	const firstLetterUppercase = (str: string) => {
		return str.charAt(0).toUpperCase() + str.slice(1)
	}

	useEffect(() => {
		ReactSession.setStoreType("sessionStorage")
		if (ReactSession.get("USER_EMAIL") === undefined) {
			navigate("/login")
		} else {
				if (ReactSession.get("USER_ROLE") === "student") {
					axios.get("http://localhost:8000/api/get_photo_list", { params: { id: ReactSession.get("USER_ID") } }).then(function (response) {
					setUserPhoto(JSON.parse(response.data.data))
				})
				axios.get("http://localhost:8000/api/get_attendance_list", { params: { id: ReactSession.get("USER_ID") } }).then(function (response) {
					setAttendanceList(JSON.parse(response.data.data))
				})
			}
		}
	}, [])

	return (
		<div className="background">
			<ProfileIconName
				name={ReactSession.get("USER_EMAIL") !== undefined ? firstLetterUppercase(ReactSession.get("USER_NAME")) + " " + firstLetterUppercase(ReactSession.get("USER_SURNAME")) : ""}
			/>
			<LogoutButton />
			{ReactSession.get("USER_ROLE") === "admin" ? <Admin /> : <div className="centralContainer">{userPhoto.length > 0 ? <Home attendanceList={attendanceList} /> : <UploadPhoto />}</div>}
		</div>
	)
}

function Home({ attendanceList, userPhoto }: { attendanceList: AttendanceList[]; userPhoto?: string[] }) {
	const navigate = useNavigate()

	return (
		<div className="infoContainer">
			<div className="leftContainer">
				<Button
					text="Add another photo!"
					img={images.selfie_emoji}
					shadow={false}
					onClick={() => {
						navigate("/add-face")
					}}
				/>
				<Button
					text="Your photos"
					img={images.many_faces_emoji}
					shadow={false}
					onClick={() => {
						navigate("/get-faces", { state: { userPhoto: userPhoto } })
					}}
				></Button>
				<Button
					text="Your details"
					img={images.details_emoji}
					shadow={false}
					onClick={() => {
						navigate("/detail")
					}}
				/>
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
					{attendanceList.map((item: any, index: number) => (
						<div key={"attendance" + index}>
							<AttendanceRow date={moment(item.date, "YYYY MM DD HH:mm:ss")} paid={item.paid} />
						</div>
					))}
				</div>
			</div>
		</div>
	)
}
function AttendanceRow({ date, paid }: AttendanceRowProps) {
	return (
		<>
			<div className="row">
				<p>{`${date.format("DD-MM-YYYY")}`}</p>
				<p>{`${date.format("HH:mm")}`}</p>
				<p>{paid % 1 !== 0 ? paid : `${paid}.00`}</p>
			</div>
		</>
	)
}
function UploadPhoto() {
	const navigate = useNavigate()

	return (
		<>
			<p>You haven't uploaded any photo yet, upload it in order to start using the recognition system!</p>
			<Button
				text={`Upload a photo of \n your face!`}
				img={images.face_emoji}
				shadow={true}
				onClick={() => {
					navigate("/add-face")
				}}
			/>
		</>
	)
}

export default Homepage
