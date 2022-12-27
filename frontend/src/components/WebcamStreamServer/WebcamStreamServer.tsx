import React, { useState, useRef, useEffect, useContext } from "react"
import Webcam from "react-webcam"
import "./WebcamStreamServer.scss"
import Resolution from "../../types/Resolution.js"
import { AdminContext } from "../../pages/Admin/Admin"

type WebcamStreamCaptureProps = {
	style?: React.CSSProperties
	resolution: Resolution
}
/**
 * Webcam components that handles the video recording
 * @param param0
 * @returns
 */
const WebcamStreamServer = ({ style, resolution }: WebcamStreamCaptureProps) => {
	const webcamRef = useRef<any>(null)
	const [connected, setConnected] = useState<boolean>(false)
	const [imgSrc, setImgSrc] = useState<any>(null)
	const [photoInterval, setPhotoInterval] = useState<any>(null)
	let { socket, setSocket } = useContext(AdminContext)

	const FPS = 15

	const periodicScreenshot = async () => {
		const photoInterval = setInterval(async () => {
			if (webcamRef.current === null) return
			const screenshot: string = webcamRef.current.getScreenshot()
			socket?.send(screenshot)
		}, 1000 / FPS)
		setPhotoInterval(photoInterval)
	}

	useEffect(() => {
		setSocket(new WebSocket(`ws://127.0.0.1:8000/ws/socket-server/`))
		return () => {
			socket?.close()
			setSocket(undefined)
		}
	}, [])

	useEffect(() => {
		socket?.addEventListener("open", () => setConnected(true))
		socket?.addEventListener("close", () => setConnected(false))
		socket?.addEventListener("message", (e: any) => {
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
