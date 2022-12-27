import React from "react"

import "./Button.scss"

type ButtonProps = {
	text: string
	img?: string
	shadow: boolean
	isActive?: boolean
	onClick: (event: React.MouseEvent<HTMLButtonElement>) => void
}

const Button = ({ text, img, shadow, isActive, onClick }: ButtonProps) => {
	const style = {
		boxShadow: shadow ? "4px 5px 19px rgba(var(--shadow-color-rgb), 0.5)" : "",
		opacity: isActive ? 1 : 0.5,
	}
	return (
		<>
			<button style={style} onClick={onClick}>
				<p>
					{/* Strange trick to use \n in the text props */}
					{text.split("\n").map((str, index) => (
						<span key={"row" + index}>{str}</span>
					))}
				</p>
				{img ? <img alt={`${img}`} src={img}></img> : <></>}
			</button>
		</>
	)
}

export default Button
