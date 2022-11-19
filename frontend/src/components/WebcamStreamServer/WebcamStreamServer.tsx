import React, { useCallback, useState, useRef, useEffect } from "react"
import Webcam from "react-webcam"
import Button from "../Button/Button"

type WebcamStreamCaptureProps = {
	style?: React.CSSProperties
	startRecordingText: string
	endRecordingText: string
	socket: WebSocket
}
/**
 * Webcam components that handles the video recording
 * Source: https://codepen.io/mozmorris/pen/yLYKzyp?editors=0010
 * @param param0
 * @returns
 */
const WebcamStreamServer = ({ style, startRecordingText, endRecordingText, socket }: WebcamStreamCaptureProps) => {
	const webcamRef = useRef<any>()
	const [capturing, setCapturing] = useState(false)
	const [imgSrc, setImgSrc] = useState<any>(null)

	const FPS = 15

	useEffect(() => {
		socket.addEventListener("message", (e: any) => {
			setImgSrc(e.data)
		})
	}, [socket])

	const handleStartCaptureClick = useCallback(async () => {
		setCapturing(true)

		periodicScreenshot()
	}, [webcamRef, setCapturing])

	const periodicScreenshot = async () => {
		setInterval(() => {
			const screenshot = webcamRef.current.getScreenshot()
			socket.send(screenshot)
		}, 1000 / FPS)
	}

	const handleStopCaptureClick = useCallback(() => {
		setCapturing(false)
	}, [setCapturing])

	return (
		<>
			<div style={{ display: "flex", flexDirection: "row" }}>
				<Webcam mirrored audio={false} ref={webcamRef} style={style} screenshotFormat="image/jpeg" />
				<Button text={capturing ? endRecordingText : startRecordingText} onClick={capturing ? handleStopCaptureClick : handleStartCaptureClick} shadow={true} />
				{imgSrc && <img alt="stremed-video" src={imgSrc} />}
			</div>
		</>
	)
}

export default WebcamStreamServer
