import React, { useCallback, useState, useRef, useEffect, useContext } from "react"
import Webcam from "react-webcam"
import Button from "../Button/Button"
import { useCountdown } from "usehooks-ts"

type WebcamStreamCaptureProps = {
	style?: React.CSSProperties
	downloadName?: string
	setUploadCompleted: (value: boolean) => void
}
/**
 * Webcam components that handles the video recording
 * Source: https://codepen.io/mozmorris/pen/yLYKzyp?editors=0010
 * @param param0
 * @returns
 */
const WebcamStreamCapture = ({ style, downloadName, setUploadCompleted }: WebcamStreamCaptureProps) => {
	const webcamRef = useRef<any>()
	const mediaRecorderRef = useRef<MediaRecorder>()
	const [capturing, setCapturing] = useState(false)
	const [recordedChunks, setRecordedChunks] = useState([])
	const [count, { startCountdown }] = useCountdown({
		countStart: 10,
		intervalMs: 1000,
	})

	useEffect(() => {
		if (count === 0) handleStopCaptureClick()
	}, [count])

	const handleStartCaptureClick = useCallback(() => {
		setCapturing(true)
		mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
			mimeType: "video/webm",
		})
		mediaRecorderRef.current.addEventListener("dataavailable", handleDataAvailable)
		mediaRecorderRef.current.start(1)
		startCountdown()
	}, [webcamRef, setCapturing, mediaRecorderRef])

	const handleDataAvailable = useCallback(
		({ data }: any) => {
			if (data.size > 0) {
				setRecordedChunks((prev) => prev.concat(data))
			}
		},
		[setRecordedChunks]
	)
	const handleDownload = useCallback(() => {
		if (recordedChunks.length) {
			const blob = new Blob(recordedChunks, {
				type: "video/webm",
			})
			const url = URL.createObjectURL(blob)
			const a = document.createElement("a")
			document.body.appendChild(a)
			a.setAttribute("display", "none")
			a.href = url
			a.download = "face_video.mp4"
			a.click()
			window.URL.revokeObjectURL(url)
			setRecordedChunks([])

			setUploadCompleted(true)
		}
	}, [recordedChunks])

	const handleStopCaptureClick = useCallback(() => {
		mediaRecorderRef.current?.stop()
		setCapturing(false)
		handleDownload()
	}, [mediaRecorderRef, setCapturing, handleDownload])

	return (
		<>
			<Webcam mirrored audio={false} ref={webcamRef} style={style} />
			<Button text={capturing ? count.toString() : "Start recording"} onClick={capturing ? () => {} : handleStartCaptureClick} shadow={true} />
		</>
	)
}

export default WebcamStreamCapture
