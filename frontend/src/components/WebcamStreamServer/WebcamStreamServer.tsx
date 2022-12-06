import React, { useState, useRef, useEffect } from "react"
import Webcam from "react-webcam"
import "./WebcamStreamServer.scss"

type WebcamStreamCaptureProps = {
	style?: React.CSSProperties
	connected: boolean
	socket: WebSocket
}
/**
 * Webcam components that handles the video recording
 * Source: https://codepen.io/mozmorris/pen/yLYKzyp?editors=0010
 * @param param0
 * @returns
 */
const WebcamStreamServer = ({ style, connected, socket }: WebcamStreamCaptureProps) => {
	const webcamRef = useRef<any>(null)
	const [imgSrc, setImgSrc] = useState<any>(null)

	const FPS = 15

	const periodicScreenshot = async () => {
		setInterval(() => {
			const screenshot = webcamRef.current.getScreenshot()
			socket.send(screenshot)
		}, 1000 / FPS)
	}

	useEffect(() => {
		socket.addEventListener("message", (e: any) => {
			setImgSrc(e.data)
		})
	}, [socket])

	useEffect(() => {
		if (connected) {
			periodicScreenshot()
		}
	}, [connected])

	return (
		<>
			<Webcam mirrored audio={false} ref={webcamRef} screenshotFormat="image/jpeg" style={style} />
			{imgSrc && connected && <img alt="stremed-video" src={imgSrc} style={{ ...style, zIndex: 5 }} />}
		</>
	)
}

export default WebcamStreamServer
