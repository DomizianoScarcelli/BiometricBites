import React from "react"

import "./Homepage.scss"
import images from "../../constants/images"

function Homepage() {
	return (
		<div className="background">
			<NameIcon />
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

const NameIcon = () => {
	return (
		<div className="container">
			<div className="profileIcon">
				<p>D</p>
			</div>
			<p>
				Good Evening, <br /> Domiziano Scarcelli!
			</p>
		</div>
	)
}

export default Homepage
