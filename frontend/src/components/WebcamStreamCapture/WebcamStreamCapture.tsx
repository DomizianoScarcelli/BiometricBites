import React, { useCallback, useState, useRef, useEffect } from "react"
import Webcam from "react-webcam"
import Button from "../Button/Button"

type WebcamStreamCaptureProps = {
	style?: React.CSSProperties
}
/**
 * Webcam components that handles the video recording
 * Source: https://codepen.io/mozmorris/pen/yLYKzyp?editors=0010
 * @param param0
 * @returns
 */
const WebcamStreamCapture = ({ style }: WebcamStreamCaptureProps) => {
	const webcamRef = useRef<any>()
	const [photoList, setPhotoList] = useState<Array<string>>([])

	const takePhoto = () => {
		const screenshot: string = webcamRef.current.getScreenshot()
		photoList.push(screenshot)
		console.log(photoList)
	}

	return (
		<>
			<Webcam mirrored audio={false} ref={webcamRef} style={style} />
			<Button text={"Take photo"} onClick={takePhoto} shadow={true} />
		</>
	)
}

export default WebcamStreamCapture
