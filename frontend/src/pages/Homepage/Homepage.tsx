import React from "react"

import "./Homepage.scss"
import images from "../../constants/images"
import ProfileIconName from "../../components/ProfileIconName/ProfileIconName"

function Homepage() {
	return (
		<div className="background">
			<ProfileIconName />
			<div className="centralContainer">
				<p>You haven't uploaded any photo yet, upload it in order to start using the recognition system!</p>
				<button>
					<p>
						Upload a photo of <br /> your face!
					</p>
					<img alt="face emoji" src={images.face_emoji}></img>
				</button>
			</div>
		</div>
	)
}

export default Homepage
