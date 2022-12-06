import React, { useEffect, useState } from "react"
import WebcamStreamServer from "../../components/WebcamStreamServer/WebcamStreamServer"
import "./Admin.scss"
import { LogoutButton } from "../../components"

export default function Admin() {
	const [socket, setSocket] = useState<WebSocket>(new WebSocket(`ws://127.0.0.1:8000/ws/socket-server/`))
	const [connected, setConnected] = useState<boolean>(false)

	const webcamStyle: React.CSSProperties = {
		textAlign: "center",
		height: "35rem",
		width: "50rem",
		objectFit: "cover",
		borderRadius: "2rem",
		position: "absolute",
		top: "50%",
		transform: "translate(-70%, -50%)",
	}

	const openSocketConnection = () => {
		const url = `ws://127.0.0.1:8000/ws/socket-server/`
		const socket: WebSocket = new WebSocket(url)
		setSocket(socket)

		socket.addEventListener("open", (e: any) => {
			setConnected(true)
			console.log("Connected to server")
		})

		socket.addEventListener("close", (e: any) => {
			setConnected(false)
			console.log("Disconnected from server")
		})
	}

	useEffect(openSocketConnection, [])

	return (
		<div className="admin-container background">
			<LogoutButton />
			<div className="admin-container__left">
				<WebcamStreamServer connected={connected} socket={socket} style={webcamStyle} />
			</div>
			<div className="admin-container__right">
				<p>Test</p>
			</div>
		</div>
	)
}
