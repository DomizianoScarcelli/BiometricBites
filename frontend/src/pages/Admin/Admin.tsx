import React, { useEffect } from "react"
import WebcamStreamServer from "../../components/WebcamStreamServer/WebcamStreamServer"

export default function Admin() {
	const testSockets = () => {
		const url = `ws://127.0.0.1:8000/ws/socket-server/`
		const socket: WebSocket = new WebSocket(url)
		socket.addEventListener("message", (e: any) => {
			const data = JSON.parse(e.data)
			console.log(data)
		})
	}

	useEffect(testSockets, [])
	return <div>{/* <WebcamStreamServer startRecordingText="Start stream" endRecordingText="End stream" /> */}</div>
}
