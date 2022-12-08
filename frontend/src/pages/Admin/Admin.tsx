import React, { useEffect, useState } from "react"
import WebcamStreamServer from "../../components/WebcamStreamServer/WebcamStreamServer"
import "./Admin.scss"
import { LogoutButton, ProfileIconName } from "../../components"
import images from "../../constants/images"
import Button from "../../components/Button/Button"
import Resolution from "../../types/Resolution"

type Person = {
	name: string
	id: string
	amountToPay: number
}

export default function Admin() {
	const [socket, setSocket] = useState<WebSocket>(new WebSocket(`ws://127.0.0.1:8000/ws/socket-server/`))
	const [connected, setConnected] = useState<boolean>(false)
	const [lastPerson, setLastPerson] = useState<Person>()

	const webcamStyle: React.CSSProperties = {
		textAlign: "center",
		height: "60vh",
		width: "auto",
		objectFit: "cover",
		borderRadius: "2rem",
	}

	const openSocketConnection = () => {
		const url = `ws://127.0.0.1:8000/ws/socket-server/`
		const socket: WebSocket = new WebSocket(url)
		setSocket(socket)

		socket.addEventListener("open", (e: any) => {
			setConnected(true)
		})

		socket.addEventListener("close", (e: any) => {
			setConnected(false)
		})
	}

	useEffect(openSocketConnection, [])

	return (
		<div className="background">
			<ProfileIconName name="Admin" />
			<div className="admin-container">
				<LogoutButton />
				<div className="admin-container__left">
					<WebcamStreamServer connected={connected} socket={socket} style={webcamStyle} resolution={Resolution.MEDIUM} />
				</div>
				<div className="admin-container__right">
					<div className="student-details">
						<h1>Domiziano Scarcelli</h1>
						<div className="student-details__inner">
							<div className="photo">
								<img alt="student_photo" src={images.photo_of_face}></img>
								<p>Accuracy: 80%</p>
							</div>

							<div className="price-to-pay">
								<p>3.00 Euro</p>
							</div>
						</div>
					</div>
					<div className="actions">
						<Button text="Ignore" shadow={true} onClick={() => {}} />
						<Button text="Pay" shadow={true} onClick={() => {}} />
					</div>
				</div>
			</div>
		</div>
	)
}
