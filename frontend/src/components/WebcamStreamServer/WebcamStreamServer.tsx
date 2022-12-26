import React, { useState, useRef, useEffect } from "react"
import Webcam from "react-webcam"
import "./WebcamStreamServer.scss"
import Resolution from "../../types/Resolution.js"

import { RecognitionInfo } from "../../pages/Admin/Admin"

type WebcamStreamCaptureProps = {
	style?: React.CSSProperties
	connected: boolean
	socket: WebSocket
	resolution: Resolution
}
/**
 * Webcam components that handles the video recording
 * Source: https://codepen.io/mozmorris/pen/yLYKzyp?editors=0010
 * @param param0
 * @returns
 */
const WebcamStreamServer = ({ style, connected, socket, resolution }: WebcamStreamCaptureProps) => {
	const webcamRef = useRef<any>(null)
	const [imgSrc, setImgSrc] = useState<any>(null)
	const [photoInterval, setPhotoInterval] = useState<any>(null)

	const FPS = 15

	const periodicScreenshot = async () => {
		const photoInterval = setInterval(async () => {
			if (webcamRef.current === null) return
			const screenshot: string = webcamRef.current.getScreenshot()
			socket.send(screenshot)
		}, 1000 / FPS)
		setPhotoInterval(photoInterval)
	}

	useEffect(() => {
		socket.addEventListener("message", (e: any) => {
			const data = JSON.parse(e.data)
			const frame = data["FRAME"]
			setImgSrc(frame)
		})
	}, [socket])

	useEffect(() => {
		if (connected) {
			periodicScreenshot()
		} else {
			if (photoInterval !== null) photoInterval.clearInterval()
		}
	}, [connected])

	return (
		<>
			{imgSrc && connected && <img alt="stremed-video" src={imgSrc} style={{ ...style, zIndex: 5 }} />}
			<Webcam mirrored audio={false} ref={webcamRef} screenshotFormat="image/jpeg" style={{ ...style, opacity: "0", height: resolution }} />
		</>
	)
}

export default WebcamStreamServer
