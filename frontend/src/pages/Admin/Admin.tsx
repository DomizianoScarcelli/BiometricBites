import React, { useEffect, useState, createContext } from "react"

import "./Admin.scss"
import { images, types } from "../../constants"
import { Button, WebcamStreamServer } from "../../components"

type Person = {
	id: string
	name: string
	surname: string
	cf: string
	toPay: number
	profileImg?: string
}

type RecognitionInfo = {
	frame: string
	recognitionPhase: boolean
	facePresent: boolean
	userInfo: Person
	similarity: number
}

interface AdminContextInterface {
	socket: WebSocket | undefined
	setSocket: (value: WebSocket | undefined) => void
}

export let AdminContext = createContext<AdminContextInterface>({
	socket: undefined,
	setSocket: () => {},
})

export default function Admin() {
	const [socket, setSocket] = useState<WebSocket>()
	const [recognitionArray, setRecognitionArray] = useState<Person[]>([])
	const [similarity, setSimilarity] = useState<number>(1)
	const [currentPerson, setCurrentPerson] = useState<Person>()
	const [similarityArray, setSimilarityArray] = useState<number[]>([])
	const [userProfilePic, setUserProfilePic] = useState(images.photo_of_face)

	const webcamStyle: React.CSSProperties = {
		textAlign: "center",
		height: "100%",
		width: "100%",
		objectFit: "cover",
		borderRadius: "2rem",
	}

	const computeAccuracy = () => {
		let startingSimilarity = 1
		for (let sim of similarityArray) {
			startingSimilarity *= sim
		}
		return startingSimilarity
	}

	const handleSocketReception = (e: any) => {
		const data: any = JSON.parse(e.data)
		const recognitionInfo: RecognitionInfo = {
			frame: data["FRAME"],
			recognitionPhase: data["RECOGNITION_PHASE"],
			facePresent: data["FACE_PRESENT"],
			userInfo: parseUserInfo(data["USER_INFO"]),
			similarity: data["SIMILARITY"],
		}
		const { recognitionPhase, facePresent, userInfo, similarity }: RecognitionInfo = recognitionInfo
		if (!recognitionPhase) return // It the frame is not used for recognition, do nothing
		if (facePresent) {
			// The face is present on the camera
			if (similarity !== undefined) setSimilarityArray((array) => array.concat(similarity))
			setRecognitionArray((array) => array.concat(userInfo))
		} else {
			setRecognitionArray([])
		}
	}

	const parseUserInfo = (userInfo: any): Person => {
		if (userInfo === null || userInfo === undefined) {
			return {
				id: "",
				name: "",
				surname: "",
				cf: "",
				toPay: 0,
			}
		}

		return {
			id: userInfo["ID"],
			name: userInfo["NAME"],
			surname: userInfo["SURNAME"],
			cf: userInfo["CF"],
			toPay: userInfo["COST"],
			profileImg: userInfo["PROFILE_IMG"],
		}
	}

	const getMostFrequentPerson = (): Person => {
		const personCounter: Map<string, number> = new Map()
		for (let person of recognitionArray) {
			const stringifiedPerson = JSON.stringify(person)
			personCounter.set(stringifiedPerson, personCounter.has(stringifiedPerson) ? personCounter.get(stringifiedPerson)! + 1 : 1)
		}
		let maxCount = -1
		let maxPerson: Person
		for (let [person, count] of Array.from(personCounter.entries())) {
			if (count > maxCount) {
				maxCount = count
				maxPerson = JSON.parse(person)
			}
		}
		return maxPerson!
	}

	useEffect(() => {
		const lastPerson = currentPerson
		const person = getMostFrequentPerson()
		if (JSON.stringify(person) === JSON.stringify(lastPerson)) return
		setCurrentPerson(person)
		if (person?.profileImg) setUserProfilePic(person.profileImg)
	}, [recognitionArray])

	useEffect(() => {
		setSimilarity(computeAccuracy)
	}, [similarityArray])

	useEffect(() => {
		socket?.addEventListener("message", handleSocketReception)
	}, [socket])

	return (
		<AdminContext.Provider value={{ socket: socket, setSocket: setSocket }}>
			<div className="admin-container">
				<div className="admin-container__main">
					<div className="admin-container__sections">
						<div className="admin-container__left">
							<WebcamStreamServer style={webcamStyle} resolution={types.resolution.MEDIUM} />
						</div>
						<div className="admin-container__right">
							<div className="student-details">
								{recognitionArray.length === 0 ? (
									<h1>Waiting for a student</h1>
								) : (
									<>
										<h1>{`${currentPerson?.name} ${currentPerson?.surname}`}</h1>
										<div className="student-details__inner">
											<div className="photo">
												<img alt="student_photo" src={userProfilePic}></img>
												<p>{`Accuracy: ${(Math.round(similarity * 100) / 100) * 100}% `}</p>
											</div>

											<div className="price-to-pay">
												<p>{`${currentPerson?.toPay} Euro`}</p>
											</div>
										</div>
									</>
								)}
							</div>
							{recognitionArray.length !== 0 && (
								<div className="actions">
									<Button text="Ignore" shadow={true} onClick={() => {}} />
									<Button text="Pay" shadow={true} onClick={() => {}} />
								</div>
							)}
						</div>
					</div>
				</div>
			</div>
		</AdminContext.Provider>
	)
}

export type { RecognitionInfo }
